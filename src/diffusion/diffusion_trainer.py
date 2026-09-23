import os
import torch
from torch_geometric.loader import DataLoader
import matplotlib

from src.dataset.datasets import dataset_hardest_pca, get_CST_retrained_dataset, get_FMNIST_dataset
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
from torchvision.transforms import transforms, InterpolationMode
from tqdm import tqdm
from src.diffusion.ddpm import DDPM
from src.models.gps import GPS
import numpy as np
from configs.parser import ArgParser
from src.geometry.create_pixel_antenna import add_reflector_to_graph, create_pixel_ant
from torch_geometric.data import Batch
from src.graph.GNN_functions import prepare_graph
from src.geometry.mesh_functions import decompose_pyg_graph
from src.geometry.saving_functions import save_graph_dict_as_stl
from src.diffusion.smooth_binarize import smooth_binarize
from src.diffusion.diffusion_utils import create_parameter_dict
from src.diffusion.unet_attention import ConditionalDenoisingUNetSmallAttentionWithConcat
from src.metrics.ff_metrics import Metrics
import torch_geometric.transforms as T
import cma


class MSELoss(nn.Module):
    def __init__(self, reduce=True):
        super(MSELoss, self).__init__()

    def forward(self, pred, target, reduce=False):
        #pred and target are of shape (batch_size, H, W)
        pred = torch.clamp(pred, -15, 10)
        target = torch.clamp(target, -15, 10)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=(1, 2))  # Compute mean over spatial dimensions for each sample
        if reduce:
            return loss.mean()  # Return the mean loss across the batch
        return loss  # Return the loss for each sample


class DiffusionTrainer:
    def __init__(self, config, denoiser_unet, simulation_model=None):
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Init denoiser and DDPM wrapper
        self.denoiser_unet = denoiser_unet
        print('number of denoiser parameters:', sum(p.numel() for p in self.denoiser_unet.parameters() if p.requires_grad))
        self.ddpm = DDPM(self.denoiser_unet, num_ts=config['T'])
        self.ddpm.to(self.device)

        # Optimizer and device setup - Adam optimizer with exponential learning rate decay
        self.ff_loss_fn = MSELoss(reduce=False)
        self.metrics = Metrics(config)
        self.optimizer = torch.optim.Adam(self.ddpm.parameters(), lr = config['lr'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma = 0.1**(1.0/config['num_epochs']))

        # create and load the simulation model (GPS+PAIS surrogate) for evaluation
        surrogate_checkpoint = config.get('surrogate_checkpoint')
        if surrogate_checkpoint is None:
            raise ValueError("config['surrogate_checkpoint'] must point to the trained GPS+PAIS forward-model .pt")
        self.model = simulation_model
        self.model.load_state_dict(torch.load(surrogate_checkpoint))
        self.compute_pe  = T.AddLaplacianEigenvectorPE(k=10, attr_name='pe')

    def train(self, train_loader, test_loader, evaluation_loader):
        batch_losses = []
        epoch_losses = []
        global_step = 0

        for epoch in tqdm(range(self.config['num_epochs'])):
            self.ddpm.train()  # Set the model to training mode
            epoch_loss = 0.0
            for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{self.config['num_epochs']}"):
                data, label, class_labels, raw_idx = self.get_batch_data(batch)
                self.optimizer.zero_grad()
                data = data.to(self.device)
                label = label.to(self.device)
                loss = self.ddpm(data, label)
                loss.backward()  # Compute gradients
                self.optimizer.step()  # Update weights
                batch_loss = loss.item()

                batch_losses.append(batch_loss)
                global_step += 1
                epoch_loss += batch_loss

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.config['num_epochs']}] - Average Loss: {avg_epoch_loss:.4f}")
            self.scheduler.step()
            epoch_losses.append(avg_epoch_loss)

            self.evaluate(evaluation_loader, epoch, batch_losses, epoch_losses)
        # save the trained model
        torch.save(self.denoiser_unet.state_dict(), os.path.join(self.config["checkpoints_path"], f"diffusion_model_{self.denoiser_unet.__class__.__name__}_epoch_{self.config['num_epochs']}_bs_{self.config['batch_size']}_lr_{self.config['lr']}.pt"))

        self.test(test_loader, guidance_scale=self.config['guidance_scale'])
        print('done')


    def test(self, eval_loader, guidance_scale = 1, number_of_best_samples = 5, save_random_samples=False):
        min_losses = []
        for idx, batch in tqdm(enumerate(eval_loader), total=len(eval_loader), desc="Testing the trained diffusion model"):
            prosessed_examples = [f.split('_')[0] for f in os.listdir(self.config['output_dir'])]
            self.ddpm.eval() # changes the behavior of BN and Dropouts layers
            with torch.no_grad():
                gt_ant_matrix, eval_labels, class_labels, raw_idx = self.get_batch_data(batch)
                if raw_idx[0] in prosessed_examples:
                    print(f"Sample {raw_idx[0]} already processed, skipping...")
                    continue
                eval_labels = eval_labels[0].expand(self.config['number_of_samples_to_generate'], -1, -1)  # repeat to match batch size

                eval_labels = eval_labels.to(self.device)
                samples = self.ddpm.sample(
                    c=eval_labels,
                    img_wh=self.config['img_wh'],
                    guidance_scale=guidance_scale,
                    in_channels=2 if self.config['data_set_type'] == 'pixel_data_with_reflectors' else 1
                )
                # Now predict farfield for all generated samples in a batch
                pred_list, gt_list, losses, antenna_list, valid_indices = self.pred_batch_ff(samples, eval_labels)

                if not losses:
                    print("No valid samples generated in this batch.")
                    continue

                if save_random_samples:
                    num_random_samples = min(number_of_best_samples, len(losses))
                    random_indices = np.random.choice(len(losses), size=num_random_samples, replace=False)
                    for rank, random_idx in enumerate(random_indices):
                        random_antenna = samples[random_idx]
                        random_graph = antenna_list[random_idx]
                        random_farfield = pred_list[random_idx]
                        gt_farfield = gt_list[random_idx]
                        self.save_diffusion_output(
                            random_graph,
                            random_antenna,
                            gt_ant_matrix[0][0],
                            random_farfield,
                            gt_farfield,
                            losses[random_idx],
                            class_labels[0],
                            f"{raw_idx[0]}_top_{rank+1}"
                        )
                else:
                    # Save the best 5 samples (lowest losses)
                    best_k = min(number_of_best_samples, len(losses))
                    best_indices = np.argsort(losses)[:best_k]
                    for rank, idx in enumerate(best_indices):
                        min_loss = losses[idx]
                        # Map back to original sample index
                        original_idx = valid_indices[idx]
                        min_antenna = samples[original_idx]
                        min_graph = antenna_list[idx]
                        min_farfield = pred_list[idx]
                        gt_farfield = gt_list[idx]
                        if rank == 0:
                            self.metrics.update(min_farfield.unsqueeze(0).unsqueeze(0), gt_farfield.unsqueeze(0).unsqueeze(0))
                        min_losses.append(min_loss)
                        # Save output for each of the best samples
                        self.save_diffusion_output(
                            min_graph,
                            min_antenna,
                            gt_ant_matrix[0][0],
                            min_farfield,
                            gt_farfield,
                            min_loss,
                            class_labels[0],
                            f"{raw_idx[0]}_top_{rank+1}"
                        )

        computed_metrics = self.metrics.compute()
        print("Evaluation Metrics over the Evaluation Set:")
        for metric_name, metric_value in computed_metrics.items():
            print(f"{metric_name.upper()}: {metric_value:.4f}")
        avg_loss = sum(min_losses) / len(min_losses)
        std_loss = np.std(min_losses)
        print(f"Average Farfield Loss over Evaluation Set: {avg_loss:.4f}")
        print(f"Standard Deviation of Farfield Loss over Evaluation Set: {std_loss:.4f}")
        print('min loss :', min(min_losses))
        print(f"Guidance Scale used: {guidance_scale}")

        print('done')


    def pred_batch_ff(self, samples, eval_labels):
        env_dict, reflectors_dict = create_parameter_dict(self.config)
        losses = []
        pred_list = []
        gt_list = []
        antenna_list = []
        valid_indices = []
        batch_size = samples.size(0)  # self.config['number_of_samples_to_generate']
        max_batch_size = 32

        # Split the evaluation batch into smaller chunks if necessary
        num_splits = (batch_size + max_batch_size - 1) // max_batch_size  # Calculate the number of splits
        for split_idx in range(num_splits):
            start_idx = split_idx * max_batch_size
            end_idx = min((split_idx + 1) * max_batch_size, batch_size)
            antenna_chunk_list = []
            chunk_valid_indices = []
            # Process each split
            for i in range(start_idx, end_idx):
                matrix = smooth_binarize(samples[i, 0], 300, 300)
                antenna, _ = create_pixel_ant(
                    matrix,  # matrix is now always a Parameter with gradients
                    threshold=self.config.get('threshold'),
                    size_of_patch_in_mm=env_dict['patch_x'],
                    size_of_FR4_in_mm=env_dict['ground_x'],
                    size_of_ground=env_dict['ground_x'],
                    height=env_dict['h'],
                    reflectors_dict=reflectors_dict,
                    create_physical_pixel_mesh=True
                )

                if self.config['data_set_type'] == 'pixel_data_with_reflectors':
                    reflector_matrix = smooth_binarize(samples[i, 1], 300, 300)
                    antenna = add_reflector_to_graph(antenna, reflector_matrix, env_dict)

                antenna, graph_dict = prepare_graph(antenna.to(self.device), self.config)

                # compute Positional Encoding for the graph and add it to the node features (if not already computed in prepare_graph):
                try:
                    antenna = self.compute_pe(antenna)
                    antenna_chunk_list.append(antenna)
                    antenna_list.append(antenna)
                    chunk_valid_indices.append(i)
                except Exception as e:
                    print(f"Skipping sample {i} due to PE error: {e}")
                    continue

            if not antenna_chunk_list:
                continue

            antenna = Batch.from_data_list(antenna_chunk_list)

            # Predict farfield using the GNN model
            self.model.eval()
            with torch.no_grad():
                pred_radiation = self.model(antenna.to(self.device), radiation_image_shape=self.config['radiation_image_shape'])
            if isinstance(pred_radiation, dict):
                pred_radiation = pred_radiation['radiation_image']
            # Ground truth and predicted radiation patterns
            gt = eval_labels[chunk_valid_indices].reshape(-1, self.config['radiation_image_shape'][0], self.config['radiation_image_shape'][1]) #.reshape(-1,64,64)
            gt = nn.functional.interpolate(gt.unsqueeze(0), size=(34, 34), mode='bilinear', align_corners=False).squeeze(0)
            pred = pred_radiation.reshape(-1, 34, 34)

            # Calculate farfield loss based on GNN prediction
            loss = self.ff_loss_fn(pred, gt)

            # Accumulate results
            losses.extend(loss.tolist())
            pred_list.extend(pred.cpu())
            gt_list.extend(gt.cpu())
            valid_indices.extend(chunk_valid_indices)

        if losses:
            print(f"Batch Farfield min Loss: {min(losses):.4f}")
        else:
            print("Batch Farfield min Loss: N/A (all samples failed)")


        return pred_list, gt_list, losses, antenna_list, valid_indices


    def evaluate(self, eval_loader, epoch, batch_losses, epoch_losses):
        self.ddpm.eval() # changes the behaior of BN and Dropouts layers
        with torch.no_grad():
            if epoch + 1 in self.config['plot_epochs']:
                # Sample a minibatch of class labels from the test set
                batch = next(iter(eval_loader))
                eval_data, eval_labels, class_labels, raw_idx = self.get_batch_data(batch)
                eval_labels = eval_labels.to(self.device)
                samples = self.ddpm.sample(
                    c=eval_labels,
                    img_wh=self.config['img_wh'],
                    guidance_scale=self.config['guidance_scale'],
                    in_channels=2 if self.config['data_set_type'] == 'pixel_data_with_reflectors' else 1
                )

                rows = self.config['eval_batch_size'] // 5
                cols = self.config['eval_batch_size'] // rows
                fig, axes = plt.subplots(rows, cols, figsize=(5, 5))
                fig.suptitle(f"Samples after {epoch+1} epochs\n Guidance Scale = {self.config['guidance_scale']}", fontsize=16)
                axes = axes.flatten()

                for i in range(self.config['eval_batch_size']):
                    axes[i].imshow(samples[i, 0].cpu(), cmap='gray')
                    axes[i].axis('off')
                    axes[i].set_title(f'Digit {class_labels[i].item()}')
                plt.tight_layout()
                plt.show()

                if self.config['data_set_type'] == 'pixel_data_with_reflectors':
                    # Plot the reflector channel as well
                    fig, axes = plt.subplots(rows, cols, figsize=(5, 5))
                    fig.suptitle(f"Reflector Channel after {epoch+1} epochs\n Guidance Scale = {self.config['guidance_scale']}", fontsize=16)
                    axes = axes.flatten()

                    for i in range(self.config['eval_batch_size']):
                        axes[i].imshow(samples[i, 1].cpu(), cmap='gray')
                        axes[i].axis('off')
                        axes[i].set_title(f'Digit {class_labels[i].item()} - Reflector')
                    plt.tight_layout()
                    plt.show()



                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                fig.suptitle(f"Guidance Scale = {self.config['guidance_scale']}", fontsize=16)
                ax[0].plot(range(len(batch_losses)), batch_losses, 'b-')
                ax[0].set_xlabel('Training Steps')
                ax[0].set_ylabel('Loss')
                ax[0].set_title('Batch Loss')

                # epoch losses
                ax[1].plot(range(len(epoch_losses)), epoch_losses, 'r-')
                ax[1].set_xlabel('Epochs')
                ax[1].set_ylabel('Loss')
                ax[1].set_title('Epoch Loss')

                plt.tight_layout()
                plt.show()


    def generate_sample(self, eval_label, sample_name: str, guidance_scale = 1, number_of_best_samples = 5, run_idx=0):
        min_losses = []
        self.ddpm.eval() # changes the behavior of BN and Dropouts layers

        gt_ant_matrix =None
        eval_labels=eval_label
        class_labels = None
        raw_idx = sample_name
        self.metrics.reset()  # Reset metrics at the start of evaluation
        with torch.no_grad():
            eval_labels = eval_labels.expand(self.config['number_of_samples_to_generate'], -1, -1)  # repeat to match batch size

            eval_labels = eval_labels.to(self.device)
            samples = self.ddpm.sample(
                c=eval_labels,
                img_wh=self.config['img_wh'],
                guidance_scale=guidance_scale,
                in_channels=2 if self.config['data_set_type'] == 'pixel_data_with_reflectors' else 1
            )
            # Now predict farfield for all generated samples in a batch


            pred_list, gt_list, losses, antenna_list, valid_indices = self.pred_batch_ff(samples, eval_labels)

            if not losses:
                print("No valid samples generated in this batch.")

            # Save the best 5 samples (lowest losses)
            best_k = min(number_of_best_samples, len(losses))
            best_indices = np.argsort(losses)[:best_k]
            for rank, idx in enumerate(best_indices):
                min_loss = losses[idx]
                # Map back to original sample index
                original_idx = valid_indices[idx]
                min_antenna = samples[original_idx]
                min_graph = antenna_list[idx]
                min_farfield = pred_list[idx]
                gt_farfield = gt_list[idx]

                if rank == 0:
                    self.metrics.update(min_farfield.unsqueeze(0).unsqueeze(0), gt_farfield.unsqueeze(0).unsqueeze(0))
                min_losses.append(min_loss)
                # Save output for each of the best samples
                self.save_diffusion_output(
                    min_graph,
                    min_antenna,
                    gt_ant_matrix[0][0] if gt_ant_matrix is not None else None,
                    min_farfield,
                    gt_farfield,
                    min_loss,
                    class_labels[0] if class_labels is not None else None,
                    f"{raw_idx}_top_{rank+1}_{run_idx}"
                )

        avg_loss = sum(min_losses) / len(min_losses) if min_losses else float('inf')
        print(f"Average Farfield Loss for generated samples: {avg_loss:.4f}")
        print(f"Guidance Scale used: {guidance_scale}")
        self.metrics.compute()
        for metric_name, metric_value in self.metrics.compute().items():
            print(f"{metric_name.upper()}: {metric_value:.4f}")
        print('done')
        return self.metrics.compute()


    def optimize_sample_with_cma(self,matrix, eval_label):
        def objective(x, target, grid_size=16):
            matrix = torch.tensor(x).reshape(-1,grid_size, grid_size).unsqueeze(0)
            pred_list, gt_list, losses, antenna_list, valid_indices = self.pred_batch_ff(matrix, target)
            if not losses:
                return float('inf')  # Return a large loss if no valid samples were generated
            loss = losses[0]
            return loss
        es = cma.CMAEvolutionStrategy(matrix.flatten().cpu(), 0.2)
        es.optimize(objective, args=(eval_label,), iterations=100)
        # Get the best solution found
        best_x = es.result.xbest          # the raw continuous vector
        best_loss = es.result.fbest       # the best loss achieved
        # Convert back to binary antenna matrix
        best_matrix = torch.tensor(best_x).reshape(-1, 16, 16).unsqueeze(0).to(self.device)
        pred_list, gt_list, losses, antenna_list, valid_indices = self.pred_batch_ff(best_matrix, eval_label)
        # plot the best solution:
        best_graph = antenna_list[0]
        min_loss = losses[0]
        # Map back to original sample index
        original_idx = valid_indices[0]
        min_antenna = best_matrix[0]
        min_graph = antenna_list[0]
        min_farfield = pred_list[0]
        gt_farfield = gt_list[0]
        return best_matrix, min_loss, min_graph, min_antenna, min_farfield, gt_farfield


    def test_sample(self, eval_loader):
        # now for the trained model
        # plot eval images for different guidance scales
        guidance_losses = []
        batch = next(iter(eval_loader))

        for guidance_scale in [0, 1, 2, 3, 4, 5, 6, 7]:
            self.ddpm.eval() # changes the behavior of BN and Dropouts layers
            with torch.no_grad():
                eval_data, eval_labels, class_labels, raw_idx = self.get_batch_data(batch)
                eval_labels = eval_labels[0].expand(self.config['eval_batch_size'], -1, -1)  # repeat to match batch size

                eval_labels = eval_labels.to(self.device)
                samples = self.ddpm.sample(
                    c=eval_labels,
                    img_wh=self.config['img_wh'],
                    guidance_scale=guidance_scale,
                    in_channels=2 if self.config['data_set_type'] == 'pixel_data_with_reflectors' else 1
                )

                rows = self.config['eval_batch_size'] // 5
                cols = self.config['eval_batch_size'] // rows
                fig, axes = plt.subplots(rows, cols, figsize=(5, 5))
                fig.suptitle(f"Guidance Scale = {guidance_scale}", fontsize=16)
                axes = axes.flatten()

                for i in range(self.config['eval_batch_size']):
                    axes[i].imshow(samples[i, 0].cpu(), cmap='gray')
                    axes[i].axis('off')
                    axes[i].set_title(f'Digit {class_labels[i].item()}')

                plt.tight_layout()
                plt.show()

                # Create a single figure for all samples
                rows = self.config['eval_batch_size'] // 5
                cols = 5
                fig, axes = plt.subplots(rows, cols * 2, figsize=(15, rows * 3))
                fig.suptitle(f"Predicted and Ground Truth Radiation Patterns for Guidance Scale = {guidance_scale}", fontsize=16)

                env_dict, reflectors_dict = create_parameter_dict(self.config)
                losses = []
                for i in range(self.config['eval_batch_size']):
                    matrix = smooth_binarize(samples[i, 0], 300, 300)
                    antenna, _ = create_pixel_ant(
                        matrix,  # matrix is now always a Parameter with gradients
                        threshold=self.config.get('threshold'),
                        size_of_patch_in_mm=env_dict['patch_x'],
                        size_of_FR4_in_mm=env_dict['ground_x'],
                        size_of_ground=env_dict['ground_x'],
                        height=env_dict['h'],
                        reflectors_dict=reflectors_dict,
                        create_physical_pixel_mesh=True
                    )
                    antenna = Batch.from_data_list([antenna])
                    self.model.eval()
                    with torch.no_grad():
                        pred_radiation = self.model(antenna.to(self.device), radiation_image_shape=self.config['radiation_image_shape'])

                    # Ground truth and predicted radiation patterns
                    gt = eval_labels[i].cpu().reshape(-1, 64, 64)
                    gt = nn.functional.interpolate(gt.unsqueeze(0), size=(34, 34), mode='bilinear', align_corners=False).squeeze(0)
                    pred = pred_radiation.cpu().reshape(34, 34)

                    # Calculate farfield loss based on GNN prediction
                    loss = self.ff_loss_fn(pred.unsqueeze(0), gt[0].unsqueeze(0), reduce=True)
                    losses.append(loss.item())

                    # Plot ground truth
                    ax_gt = axes[i // cols, (i % cols) * 2]
                    im_gt = ax_gt.imshow(gt[0], cmap='jet')
                    ax_gt.set_title(f"Ground Truth (Sample {i})", fontsize=10)
                    ax_gt.axis('off')

                    # Plot predicted radiation pattern
                    ax_pred = axes[i // cols, (i % cols) * 2 + 1]
                    im_pred = ax_pred.imshow(pred, cmap='jet')
                    ax_pred.set_title(f"Predicted (Loss: {loss.item():.4f})", fontsize=10)
                    ax_pred.axis('off')

                    # Add colorbars
                    fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
                    fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

                # Adjust layout to prevent overlapping
                plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
                avg_loss = sum(losses) / len(losses)
                plt.show()
                print(f"Average Farfield Loss for Guidance Scale {guidance_scale}: {avg_loss:.4f}")

                # Store the guidance loss for this scale
                guidance_losses.append(avg_loss)
        # Plot guidance scale vs farfield loss
        avg_guidance_loss = sum(guidance_losses) / len(guidance_losses)
        min_guidance_loss = min(guidance_losses)
        print(f"Average Guidance Loss: {avg_guidance_loss:.4f}")
        plt.figure(figsize=(8, 5))
        plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 10, 15], guidance_losses, marker='o')
        plt.title('Guidance Scale vs Farfield Loss')
        plt.xlabel('Guidance Scale')
        plt.ylabel('Average Farfield Loss')
        plt.grid()
        plt.show()

    def plot_antenna_and_farfield(self, antenna, pred_farfield, gt_farfield, title_suffix='',vmin=0, vmax=10):
        plt.figure(figsize=(15, 10))

        # Plot antenna
        plt.subplot(1, 3, 1)

        if self.config['data_set_type'] == 'pixel_data_with_reflectors':
            combined_image = torch.cat([antenna[0], antenna[1]], dim=1)  # Combine antenna and reflector channels for visualization
            combined_image = combined_image.cpu().numpy().squeeze()  # Convert to numpy and remove channel dimension
            plt.imshow(combined_image, cmap='gray')
            antenna0 = antenna[0].detach().cpu()
            if antenna0.dim() > 2:
                antenna0 = antenna0.squeeze()
            max_idx = torch.argmax(antenna0).item()
            max_row, max_col = divmod(max_idx, antenna0.shape[-1])
            plt.scatter(max_col, max_row, c='red', s=30)
        else:
            plt.imshow(antenna, cmap='gray')

        plt.title(f"Antenna {title_suffix}")
        plt.axis('off')

        # Plot farfield
        plt.subplot(1, 3, 2)
        plt.imshow(pred_farfield.reshape(34, 34), cmap='jet', vmin=vmin, vmax=vmax)
        plt.title(f"Farfield {title_suffix}")
        plt.colorbar()
        plt.axis('off')

        # Plot ground truth farfield
        plt.subplot(1, 3, 3)
        plt.imshow(gt_farfield.reshape(34, 34), cmap='jet', vmin=vmin, vmax=vmax)
        plt.title(f"Ground Truth Farfield {title_suffix}")
        plt.colorbar()
        plt.axis('off')



        plt.tight_layout()
        plt.show()
        return plt


    def get_batch_data(self, batch):
        graph,  s11, farfeilds, graph_identifier, example_paramters,  raw_idx = batch
        data = example_paramters['ant_parameters'][0].unsqueeze(1) #lets get the antenna img matrixis
        data = smooth_binarize(data, 300, 300)
        if self.config['data_set_type'] == 'pixel_data_with_reflectors':
            reflector_matrix = example_paramters['reflector_matrix'].unsqueeze(1) # get the reflector img matrix
            reflector_matrix = smooth_binarize(reflector_matrix, 300, 300)
            data = torch.cat([data, reflector_matrix], dim=1)  # concatenate along the channel dimension

        label = farfeilds#.reshape(data.shape[0],-1)  # get the farfeild image at 2400 MHz
        class_labels = example_paramters['class_label']  # get the class labels
        return data, label, class_labels, raw_idx

    def save_diffusion_output(self, min_graph, min_ant_matrix, gt_ant_matrix, min_farfield, gt_farfield,min_loss, class_label, raw_idx):
        os.makedirs(self.config['output_dir'], exist_ok=True)
        save_path = os.path.join(self.config['output_dir'], f'{raw_idx}')
        os.makedirs(save_path, exist_ok=True)
        antenna_graph_dict = decompose_pyg_graph(min_graph.to(min_ant_matrix.device))
        save_graph_dict_as_stl(antenna_graph_dict, output_dir=save_path)
        plt = self.plot_antenna_and_farfield(min_ant_matrix.cpu(), min_farfield, gt_farfield, title_suffix=f"(Sample {raw_idx}, Loss: {min_loss:.4f})")
        plt.savefig(os.path.join(save_path, f'antenna_and_farfield_idx_{raw_idx}_class_label_{class_label}.png'))
        plt.close()
        # save gt and pred farfield as pt
        torch.save(min_farfield, os.path.join(save_path, f'pred_farfield.pt'))
        torch.save(gt_farfield, os.path.join(save_path, f'gt_farfield.pt'))
        torch.save(min_ant_matrix, os.path.join(save_path, f'pred_antenna_matrix.pt'))
        torch.save(gt_ant_matrix, os.path.join(save_path, f'gt_antenna_matrix.pt'))
        print(f"Diffusion samples saved to {save_path}")


def seed_everything(seed):
  torch.cuda.manual_seed(seed)
  torch.manual_seed(seed)

common_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),        # ensure single channel
        transforms.Resize((16, 16), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),                              # -> [0,1]
        transforms.Lambda(lambda t: t.clamp(0.0, 1.0)),     # just to be explicit
    ])

if __name__ == "__main__":
    YOUR_SEED = 0 # modify if you want
    seed_everything(YOUR_SEED)

    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    arg_parser = ArgParser(config_file=os.path.join(_REPO_ROOT, 'configs', 'inverse', 'diffusion.yaml'))
    config = arg_parser.get_config()
    # Defaults below are exactly the values used for the paper's runs; any of
    # them can be overridden from the yaml config or the command line
    # (setdefault: yaml/CLI win over these inline defaults).
    _defaults = {
        'guidance_scale': 1.0,  # CFG w (Table 2); Fig. 3 uses manual_targets_guidance_scale
        'plot_epochs':  [1,10, 15, 20],
        'num_hidden': 128,
        'batch_size': 128,
        'num_epochs': 1000, #1000, #500, #1500, #400
        'lr': 1e-4, #1e-3,
        'img_wh': (16, 16),
        'eval_batch_size': 20,
        'number_of_best_samples': 5,
        'number_of_samples_to_generate': 500, # 500, #20, #100
        'T': 700,
        'num_classes': 4096,
        'idx_freq': 3,  # 5600 MHz
        'raw_node_feture_size': 17,
        'radiation_image_shape': [34, 34, 1],
        'threshold': 0.5,
        'physical_antenna': True,
        'use_node_probs': False,
        'add_sphere': False,
        'merge_duplicate_nodes': True,
        'frequencies' : [2400, 2800, 5200, 5600,6000],
        'train_freq_idxs': [0, 1, 2, 3, 4], # indices of frequencies to use for training
        'frequencies_in_scale': [2.4e9, 2.8e9, 5.2e9, 5.6e9, 6.0e9],
        'data_set_type': 'pixel_data_with_reflectors',
        'add_radius_graph_edges': True,
        # run-mode flags (set these in the yaml per experiment):
        'test_dataset': False,        # best-of-N over the 100-hardest targets (Tables 2/3)
        'test_sample': True,          # hand-designed targets in manual_targets_dir (Fig. 3)
        'save_random_samples': True,  # save 5 random samples instead of the surrogate top-5 ('Diff. - 5 rand.')
        # GPS surrogate:
        'predict_surface_current': True,
    }
    for _k, _v in _defaults.items():
        config.setdefault(_k, _v)
    # All runtime paths come from the yaml config:
    #   data_root                : dataset root (see src/dataset/datasets.py corpus layout)
    #   split_dir                : directory holding the split .pth files
    #   surrogate_checkpoint     : trained GPS+PAIS forward-model .pt (scoring)
    #   diffusion_checkpoint     : trained diffusion .pt (when pretrained_diffusion_model is True)
    #   checkpoints_path         : where training saves diffusion checkpoints
    #   output_dir               : where candidates / STL / plots are written
    #   manual_targets_dir       : hand-designed target far-fields (Fig. 3)
    #   cst_retrained_dataset_dirs (optional list): CST-retrained dataset roots to append to the train set
    for required_key in ('data_root', 'split_dir', 'surrogate_checkpoint', 'checkpoints_path', 'output_dir'):
        if config.get(required_key) is None:
            raise ValueError(f"config['{required_key}'] must be set in the yaml config file")
    os.makedirs(config["checkpoints_path"], exist_ok=True)
    os.makedirs(config["output_dir"], exist_ok=True)

    # dataset and dataloader setup:

    pe_transform = T.Compose([
        T.ToUndirected(),
        T.AddLaplacianEigenvectorPE(k=10, attr_name='pe')
    ])

    combined_dataset, train_dataset, test_dataset = dataset_hardest_pca(config, N=config.get('hardest_n', 100), pe_transform=pe_transform, dataset_name='FMNIST_CIFAR')
    FMNIST_dataset = get_FMNIST_dataset(config, pe_transform=pe_transform)
    CST_retrained_datasets = []
    for cst_retrained_dir in config.get('cst_retrained_dataset_dirs') or []:
        CST_retrained_datasets.append(get_CST_retrained_dataset(cst_retrained_dir, config, pe_transform=pe_transform))

    train_dataset = torch.utils.data.ConcatDataset([train_dataset, *CST_retrained_datasets, FMNIST_dataset])
    print(f"Train dataset size: {len(train_dataset)}")

    num_cpus = os.cpu_count() or 4
    num_workers = min(8, max(1, num_cpus // 2))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],  follow_batch=['pos_surface_current'], shuffle=True,
                                        num_workers=num_workers,
                                        pin_memory=True,
                                        persistent_workers=True,
                                        prefetch_factor=2,
                                        drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, follow_batch=['pos_surface_current'], shuffle=False)
    evaluation_loader = DataLoader(test_dataset, batch_size=config.get('eval_batch_size'), follow_batch=['pos_surface_current'], shuffle=False)

    run_seed = int.from_bytes(os.urandom(8), 'big') % (2**31 - 1)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # the models:
    simulation_model  = GPS(config=config,in_channels=16, channels=128, pe_dim=10, num_layers=10, attn_type='multihead', attn_kwargs = {'dropout': 0.3}, out_channels=config.get('radiation_image_shape')[0]*config.get('radiation_image_shape')[1]).to(device) #attn_type='performer', attn_kwargs={'num_random_features': 64})

    denoiser_unet = ConditionalDenoisingUNetSmallAttentionWithConcat(config=config, in_channels=2 , num_hiddens=config['num_hidden'], ff_in_ch=1)

    if config.get('pretrained_diffusion_model') is True:
        path_to_denoiser = config.get('diffusion_checkpoint')
        if path_to_denoiser is None:
            raise ValueError("config['diffusion_checkpoint'] must be set when 'pretrained_diffusion_model' is True")
        denoiser_unet.load_state_dict(torch.load(path_to_denoiser, map_location=device))

    trainer = DiffusionTrainer(config, denoiser_unet ,simulation_model)
    if config.get('pretrained_diffusion_model') is False: # no need to train if we are loading a pretrained model
        trainer.train(train_loader, test_loader, evaluation_loader)

    if config.get('test_dataset', True):
        if config.get('save_random_samples') is True:
            trainer.test(test_loader, guidance_scale=config['guidance_scale'], number_of_best_samples=config['number_of_best_samples'], save_random_samples=True)
        else:
            trainer.test(test_loader, guidance_scale=config['guidance_scale'], number_of_best_samples=config['number_of_best_samples'])

    if config.get('test_sample', False):
        # Hand-designed target far-fields (Fig. 3): every .pt file in
        # config['manual_targets_dir'] is treated as one target far-field.
        config['guidance_scale'] = config.get('manual_targets_guidance_scale', 2.0)
        path_to_samples_dir = config.get('manual_targets_dir')
        if path_to_samples_dir is None:
            raise ValueError("config['manual_targets_dir'] must be set when 'test_sample' is True")

        avrage_metrics ={'mse': 0, 'mae': 0, 'psnr': 0, 'ssim': 0, 'mssim': 0, 'peak_power': 0}
        for sample_name in os.listdir(path_to_samples_dir):
            idx=0
            if not sample_name.endswith('.pt'):
                continue
            config['output_dir']  = os.path.join(path_to_samples_dir, 'diffusion_outputs/')
            path_to_sample = os.path.join(path_to_samples_dir, sample_name)
            test_sample = torch.load(path_to_sample,weights_only=False)
            test_sample = torch.tensor(test_sample, dtype=torch.float32)

            # set a different seed for each sample generation
            sample_seed = int.from_bytes(os.urandom(8), 'big') % (2**31 - 1)
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)


            log_scale = False
            if log_scale:
                #return to linear scale
                linear_image = torch.pow(10, test_sample / 10)
                # plot the linear image with fixed color scale
                plt.imshow(linear_image.cpu(), cmap='jet', vmin=0, vmax=10)
                plt.title(f"Linear Scale Farfield for Sample {sample_name}")
                plt.colorbar()
                plt.show()

                test_sample = linear_image
            metrics = trainer.generate_sample(test_sample, sample_name, guidance_scale=config['guidance_scale'], number_of_best_samples=config['number_of_best_samples'], run_idx=idx)
            # save the nearest neighbor metrics to a text file
            with open(os.path.join(config['output_dir'], sample_name + '_diffusion_metrics.txt'), 'w') as f:
                f.write(f"Metrics for the nearest neighbor example:\n")
                for key, value in metrics.items():
                    f.write(f"  {key}: {value:.6f}\n")
                for key, value in metrics.items():
                    avrage_metrics[key] += value

        num_samples = len([name for name in os.listdir(path_to_samples_dir) if name.endswith('.pt')])
        print(f"Avrage metrics across {num_samples} examples:")
        for key, value in avrage_metrics.items():
            avrage_metrics[key] = value / num_samples
            print(f"  {key}: {avrage_metrics[key]:.6f}")

    trainer.test_sample(test_loader)
