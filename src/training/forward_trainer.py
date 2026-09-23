from src.losses.losses import FarfeildLoss, S11SingleFreqLoss, compute_metrics
import torch
from tqdm import tqdm
import wandb
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from src.utils import scale_to_log


class Trainer():
    def __init__(self, config, model, loss_function, optimizer, scheduler, dataset=None, graph_stats=None, ff_stats=None):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.init_peramters()
        self.ff_stats = ff_stats
        self.graph_stats = graph_stats

        self.model = model.to(self.device)

        # loss and optimizer ####################################
        self.loss_function = loss_function
        self.ff_loss_function = FarfeildLoss(self.config, self.ff_stats)
        self.s11_loss_function = S11SingleFreqLoss(self.config)

        self.learning_rate = self.config.get('learning_rate')
        self.num_epochs = config.get('total_epochs')

        self.optimizer = optimizer
        self.scheduler = scheduler
        ######################################

        self.init_wandb()


    def init_peramters(self):

        # loss and optimizer
        self.learning_rate = self.config.get('learning_rate')
        self.weight_decay = self.config.get('weight_decay')
        self.gamma = self.config.get('gamma')
        self.frequencies = self.config.get('frequencies')

        # paramters:
        self.k_sphere_neighbores = self.config.get('k_sphere_neighbores')
        self.num_epochs = self.config.get('num_epochs')
        self.raw_node_feture_size = self.config.get('raw_node_feture_size')
        nearest_neighbor_alg = 'calc_nn_from_CST_params'
        data_set_type = self.config.get('data_set_type')
        self.gnn_out_feature_size = self.config.get('gnn_out_feature_size')
        self.scale_to_DB = self.config.get('output_in_DB')
        # wandb:
        self.wandb_log_dict = {}


    def train(self, train_loader=None, test_loader=None):
        best_test_loss = float('inf')
        # Define a file path for saving the model's state_dict
        cwd = os.getcwd()
        os.makedirs(os.path.join(cwd, 'trained_models'), exist_ok=True)
        model_weights_path = cwd + '/trained_models/' + self.wandb_run.name + self.config.get('path_to_save_model_weights')

        # resume: load BEFORE the loop and use the returned epoch count to set the
        # loop's start point. (Previously this reassigned the `epoch` loop variable
        # from inside a `for epoch in range(self.num_epochs)` loop, which has no
        # effect on which values the loop actually yields next -- resuming would
        # restore model/optimizer/scheduler state correctly but then still run a
        # full self.num_epochs *more* iterations relabeled from 1, silently
        # overwriting earlier interval checkpoints and never stopping at a true
        # total of self.num_epochs.)
        start_epoch = 0
        if self.config.get('load_from_checkpoint'):
            checkpoint_path = os.path.join(self.config['checkpoints_path'], f'checkpoint.pth')
            start_epoch, _ = self.load_checkpoint(path=checkpoint_path)
            print(f'[resume] resuming training from epoch {start_epoch} (of {self.num_epochs} total)')

        # --- epoch-0 raw-state_dict checkpoint, saved before any optimizer step.
        # Independent of the best-on-improvement save above and the every-epoch
        # resume checkpoint.pth saved inside the loop below. Uses a bare state_dict
        # (not the dict-of-state-dicts save_checkpoint() format) so it can be loaded
        # unchanged by a plain load_state_dict()-based evaluation script.
        interval_dir = self.config.get('interval_checkpoint_dir')
        if interval_dir and not self.config.get('load_from_checkpoint'):
            os.makedirs(interval_dir, exist_ok=True)
            epoch0_path = os.path.join(interval_dir, 'checkpoint_epoch_0.pt')
            torch.save(self.model.state_dict(), epoch0_path)
            print(f'[interval checkpoint] saved {epoch0_path}')

        for epoch in tqdm(range(start_epoch, self.num_epochs), desc="Training GNN ",
                           unit="epoch", initial=start_epoch, total=self.num_epochs, leave=True):
            train_losses = []
            train_accuracy = []
            train_accuracy_0_4 = []
            train_surface_current_losses = []
            test_loss = 0

            # train the network:
            self.model.train()
            for idx, (graph, s11, ff_images, graph_identifier, params_vector, raw_idx) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Examples in training", unit=' examples', leave=True):
                predictions, loss_metrics = self.train_step([graph, s11, ff_images, graph_identifier, raw_idx])
                train_losses.append(loss_metrics['total_batch_loss'])
                train_accuracy.append(loss_metrics['train_batch_accuracy'])
                train_accuracy_0_4.append(loss_metrics['train_batch_accuracy_0_4'])
                train_surface_current_losses.append(loss_metrics['surface_current_loss'])

            train_epoch_loss = sum(train_losses) / len(train_losses)
            train_epoch_accuracy = sum(train_accuracy) / len(train_accuracy)
            train_epoch_accuracy_0_4 = sum(train_accuracy_0_4) / len(train_accuracy_0_4)
            train_epoch_surface_current_loss = sum(train_surface_current_losses) / len(train_surface_current_losses)
            self.scheduler.step()
            # Print current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:03d} | LR: {current_lr:.6f}")

            # test the network:
            self.model.eval()
            test_loss, test_metrics, test_accuracy, test_accuracy_0_4 = self.test_network(test_loader)

            # Check if the current validation loss is the best so far
            if test_loss < best_test_loss:
                best_test_loss = test_loss

                # Save the model's state_dict to the file
                torch.save(self.model.state_dict(), model_weights_path)
                print(f'Epoch {epoch}: Validation current best Loss: {test_loss}')

            self.validate(test_loader)
            self.wandb_log_dict.update({
                "GNN Train Loss:": train_epoch_loss,
                "GNN Test Loss:": test_loss,
                'Learning rate': self.optimizer.param_groups[0]['lr'],
                'Train accuracy (loss under 0.1)': train_epoch_accuracy,
                'Test accuracy (loss under 0.1)': test_accuracy,
                'Train percentage of exampals with loss over 0.4 ': train_epoch_accuracy_0_4,
                'Test percentage of exampals with loss over 0.4': test_accuracy_0_4,
                'Train Surface Current Loss': train_epoch_surface_current_loss,
            })
            self.wandb_run.log(self.wandb_log_dict)
            if self.config['checkpoints_path'] is not None:
                checkpoint_path = os.path.join(self.config['checkpoints_path'], f'checkpoint.pth')
                self.save_checkpoint(self.model, self.optimizer, self.scheduler, epoch+1, train_epoch_loss, path=checkpoint_path)

            # --- every-10th-epoch raw-state_dict checkpoint (separate from the two
            # mechanisms above; see epoch-0 comment for rationale).
            if interval_dir and (epoch + 1) % 10 == 0:
                interval_path = os.path.join(interval_dir, f'checkpoint_epoch_{epoch + 1}.pt')
                torch.save(self.model.state_dict(), interval_path)
                print(f'[interval checkpoint] saved {interval_path}')


    def train_step(self, batch):
        """Performs a single training step on a batch of data.

        Args:
            batch: Tuple containing (graph, s11, ff_images, graph_identifier, raw_idx)

        Returns:
            tuple: (predictions_dict, loss_metrics_dict) containing the model predictions
                   and computed loss metrics
        """
        batch_graph, s11, ff_images, graph_identifier, raw_idx = batch

        # Initialize metrics
        example_loss = 0.0
        self.optimizer.zero_grad()
        if self.config['gps']:
            self.model.redraw_projection.redraw_projections()
        pred = self.model(batch_graph.to(self.device), radiation_image_shape=self.config.get('radiation_image_shape'))
        if self.config.get('predict_surface_current', False):
            if self.config.get('use_physics_loss', False):
                ff_pred = pred['radiation_image']
                surface_current_pred = pred['surface_current']
                ff_from_sc = pred['radiation_image_physics'].unsqueeze(1)  # [B, 1, H, W]
                ff_label = ff_images.to(self.device)
                ff_pred = ff_pred.permute(0, 3, 1, 2)
                loss, ff_loss_component, surface_current_loss_component = self.loss_function(batch_graph, surface_current_pred, ff_pred, ff_from_sc, ff_label)
            else:
                ff_pred = pred['radiation_image']
                surface_current_pred = pred['surface_current']
                ff_label = ff_images.to(self.device)
                ff_pred = ff_pred.permute(0, 3, 1, 2)
                loss, ff_loss_component, surface_current_loss_component = self.loss_function(batch_graph, surface_current_pred, ff_pred, ff_label)
        elif self.config.get('position_auxiliary_task', False):
            ff_pred = pred['radiation_image']
            position_pred = pred['positions']
            ff_label = ff_images.to(self.device)
            ff_pred = ff_pred.permute(0, 3, 1, 2)
            position_label = batch_graph.pos
            loss_ff = self.ff_loss_function(ff_pred, ff_label)
            loss_position = torch.nn.functional.mse_loss(position_pred, position_label)
            loss = loss_ff + 0.1 * loss_position
            surface_current_loss_component = 0.0
        else:
            ff_label = ff_images.to(self.device)
            pred = pred.permute(0, 3, 1, 2)
            loss = self.loss_function(pred, ff_label)
            surface_current_loss_component = 0.0

        if self.config.get('pred_s11_single_freq', False):
            s11_label = batch_graph.s11['s11_complex'].to(self.device)
            s11_pred = pred['s11_pred']
            s11_loss = self.s11_loss_function(s11_pred, s11_label)
            loss += self.config.get('s11_loss_weight', 0.2) * s11_loss

        loss.backward()
        self.optimizer.step()

        # Compute final metrics
        total_batch_loss = loss.item()
        batch_accuracy = 1 if loss <= 0.1 else 0
        batch_accuracy_0_4 = 1 if loss > 0.4 else 0
        sc_loss_val = surface_current_loss_component.item() if hasattr(surface_current_loss_component, 'item') else surface_current_loss_component

        # Package results
        loss_metrics = {
            'ff_loss': loss,
            's11_loss': 0,  # Not currently used
            'total_batch_loss': total_batch_loss,
            'train_batch_accuracy': batch_accuracy,
            'train_batch_accuracy_0_4': batch_accuracy_0_4,
            'surface_current_loss': sc_loss_val,
        }

        return {}, loss_metrics  # Empty predictions dict for now

    def preflight(self, train_loader, max_steps=5):
        """Runs exactly max_steps raw training steps (reusing train_step()), with a
        per-step finite-loss check, then returns. Does NOT run epoch-level validation,
        test_network, checkpoint saving, or wandb epoch logging -- intended as a fast
        checkpoint/config sanity check before committing to a full run."""
        self.model.train()
        step_losses = []
        data_iter = iter(train_loader)
        for step in range(max_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            graph, s11, ff_images, graph_identifier, params_vector, raw_idx = batch
            _, loss_metrics = self.train_step([graph, s11, ff_images, graph_identifier, raw_idx])
            loss_value = loss_metrics['total_batch_loss']
            is_finite = (loss_value == loss_value) and (loss_value not in (float('inf'), float('-inf')))
            print(f'[preflight] step {step + 1}/{max_steps}  loss={loss_value:.6f}  finite={is_finite}')
            if not is_finite:
                raise RuntimeError(f'[preflight] non-finite loss at step {step + 1}: {loss_value}')
            step_losses.append(loss_value)
        print(f'[preflight] completed {max_steps} steps. mean_loss={sum(step_losses) / len(step_losses):.6f}')
        return step_losses

    def test_network(self, test_loader=None):
        self.model.eval()
        test_losses = []
        test_accuracy = 0
        test_accuracy_0_4 = 0
        test_surface_current_losses = []
        test_s11_complex_losses = []
        test_s11_abs_db_losses = []
        test_s11_abs_db_mae_losses = []
        # Initialize accumulators for metrics
        total_metrics = {'MAE': 0.0, 'MSE': 0.0, 'max_error': 0.0, 'SNR': 0.0, 'mssim': 0.0}
        num_of_abslute_examples = 0
        for idx, (data) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Examples in testing", unit=' examples', leave=True):

            with torch.no_grad():
                batch_graph, s11, ff_images, graph_identifier, params_vector, raw_idx = data
                pred = self.model(batch_graph.to(self.device), radiation_image_shape=self.config.get('radiation_image_shape'), is_training=False)
                if isinstance(pred, dict):
                    if self.config.get('predict_surface_current', False):
                        surface_current_pred = pred['surface_current']
                        sc_loss = self.loss_function.surface_current_loss_function(batch_graph.to(self.device), surface_current_pred)
                        test_surface_current_losses.append(sc_loss.item())

                    if self.config.get('pred_s11_single_freq', False):
                        s11_pred = pred['s11_pred']
                        s11_label = batch_graph.s11['s11_complex'].to(self.device)
                        s11_loss_dict = self.s11_loss_function(s11_pred, s11_label, return_db_loss=True)
                        s11_loss, s11_loss_db, s11_loss_db_mae = s11_loss_dict['complex_loss'], s11_loss_dict['db_loss'], s11_loss_dict['db_loss_mae']
                        test_s11_complex_losses.append(s11_loss.item())
                        test_s11_abs_db_losses.append(s11_loss_db.item())
                        test_s11_abs_db_mae_losses.append(s11_loss_db_mae.item())
                    pred = pred['radiation_image']
                if torch.isnan(pred).any():
                    print(f"NaN detected in predictions at index {idx}")
                    continue
                ff_label = ff_images.to(self.device)
                pred = pred.squeeze(-1).unsqueeze(0)
                ff_loss = self.ff_loss_function(pred, ff_label, train=False)
                example_metrics = compute_metrics(pred, ff_label, self.ff_stats)

                for key, value in example_metrics.items():
                    total_metrics[key] += value
            num_of_abslute_examples += 1

            if ff_loss <= 0.1:
                test_accuracy += 1
            if ff_loss > 0.4:
                test_accuracy_0_4 += 1

            test_losses.append(ff_loss.item())

        test_metrics = {key: value / num_of_abslute_examples for key, value in total_metrics.items()}
        self.wandb_log_dict.update(test_metrics)
        print('test_metrics: ', test_metrics)
        test_loss = sum(test_losses)/len(test_losses)
        test_accuracy = test_accuracy / len(test_losses)
        test_accuracy_0_4 = test_accuracy_0_4 / len(test_losses)
        loss_hist_image = self.loss_list_to_hist_image(test_losses)
        self.wandb_log_dict.update({"farafeild loss histogram dB-dB:": loss_hist_image})
        if test_surface_current_losses:
            self.wandb_log_dict.update({"Test Surface Current Loss": sum(test_surface_current_losses) / len(test_surface_current_losses)})
        if test_s11_complex_losses:
            self.wandb_log_dict.update({"Test S11 Complex Loss": sum(test_s11_complex_losses) / len(test_s11_complex_losses)})
        if test_s11_abs_db_losses:
            self.wandb_log_dict.update({"Test S11 Abs DB Loss": sum(test_s11_abs_db_losses) / len(test_s11_abs_db_losses)})
        if test_s11_abs_db_mae_losses:
            self.wandb_log_dict.update({"Test S11 Abs DB MAE Loss": sum(test_s11_abs_db_mae_losses) / len(test_s11_abs_db_mae_losses)})
        return test_loss, test_metrics, test_accuracy, test_accuracy_0_4


    def init_wandb(self):
        wandb_config = {
            "learning_rate": self.config.get('learning_rate'),
            "architecture": "antenna GNN farfeild directivty",
            "dataset": "Alpha datset",
            "epochs": self.config.get('num_epochs')
        }
        wandb_config.update(self.config)  # Merge with existing config

        # start a new wandb run to track this script
        self.wandb_run = wandb.init(
            # set the wandb project where this run will be logged
            project="antenna GNN farfeild directivty",
            # track hyperparameters and run metadata
            config=wandb_config
        )
        self.wandb_log_dict = {}


    def validate(self, test_loader):
        # Convert loader to a list to grab random samples
        data_list = list(test_loader)
        # Grab random examples
        random.seed(0)
        random_samples = random.sample(data_list, self.config.get('number_of_validation_samples'))
        for example_number, (batched_graph, s11, ff_labels, graph_identifier, params_vector, raw_idx) in enumerate(random_samples):
            image_list = []
            with torch.no_grad():

                raw_idx = str(raw_idx)
                pred = self.model(batched_graph.to(self.device), radiation_image_shape=self.config.get('radiation_image_shape'), is_training=False)
                if isinstance(pred, dict):
                    pred = pred['radiation_image']
                ff_label = ff_labels.to(self.device)
                pred = pred.squeeze(-1).unsqueeze(0)
                ff_loss = self.ff_loss_function(pred, ff_label, train=False)
                if ff_label.shape != pred.shape:
                    ff_label = torch.nn.functional.interpolate(ff_label, size=pred.shape[2:], mode='bilinear', align_corners=False)
                wandb_pred_image = self.tensor_to_image(pred[0], raw_idx, batched_graph.freq_hz, str_type='pred dB', mse_loss=ff_loss)
                image_list.append(wandb_pred_image)
                wandb_target_image = self.tensor_to_image(ff_label[0], raw_idx, batched_graph.freq_hz, str_type='target dB')
                image_list.append(wandb_target_image)

                s11_magnitude_label = s11['s11_abs']
                s11_magnitude_label = scale_to_log(s11_magnitude_label)
                s11_frequancies = s11['frequancy']
                wandb_s11_magnitude = self.s11_abs_to_image(s11_magnitude_label, s11_frequancies, raw_idx, 'dB')
                image_list.append(wandb_s11_magnitude)

                self.wandb_log_dict.update({"example :" + str(example_number): image_list})


    def s11_abs_to_image(self, s11_abs, frequencies, raw_idx, str_type):
        plt.figure(figsize=(25, 25))
        plt.plot(frequencies[0].cpu(), s11_abs[0].cpu())
        plt.title('Absolute Value of S11 Parameters ' + str_type)
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('|S11|')
        plt.grid(True)
        wandb_image = wandb.Image(plt, caption=str_type + "S11 Parameters' " + raw_idx + ":")
        plt.close()
        return wandb_image


    def tensor_to_image(self, tensor_image, raw_idx, frequency, str_type='linear', mse_loss=None):
        plt.figure(figsize=(25, 25))
        # Ensure the tensor is squeezed to fit the expected shape for plt.imshow
        tensor = tensor_image[0].clone().detach().cpu()  # Shape becomes [64, 64]
        plt.imshow(tensor, cmap='jet', vmin=max(self.ff_stats['min'], -15), vmax=self.ff_stats['max'] + 0.1)

        plt.colorbar()
        plt.title('Tensor Visualization')
        caption = f"farafeild' {raw_idx}:\nfrequency = {frequency} {str_type}"
        if mse_loss:
            caption += f"\nMSE: {mse_loss:.4f}"
        wandb_image = wandb.Image(plt, caption=caption)
        plt.close()
        return wandb_image


    def loss_list_to_hist_image(self, losses):
        # Create a histogram
        losses_cpu = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in losses]
        plt.hist(losses_cpu, bins=150, edgecolor='black')

        # Add labels and title
        plt.xlabel('Loss')
        plt.ylabel('Frequency')
        plt.title('Loss MSE Histogram dB')
        wandb_image = wandb.Image(plt, caption="farafeild loss histogram ")
        plt.close()
        return wandb_image


    def cheek_if_example_in_test_set(self):
        train_raw_idx_list = []
        test_raw_idx_list = []
        for graph, s11, ff_images, graph_identifier, params_vector, raw_idx in self.train_loader:
            train_raw_idx_list.append(raw_idx)

        for graph, s11, ff_images, graph_identifier, params_vector, raw_idx in self.test_loader:
            test_raw_idx_list.append(raw_idx)

        for raw_idx in test_raw_idx_list:
            if raw_idx in train_raw_idx_list:
                print('train examples exist in test set !!!')
                return True

        return False

    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss, path="checkpoint.pth"):
        """
        Save training checkpoint including model, optimizer, and scheduler state.

        Args:
            model: torch.nn.Module — your model.
            optimizer: torch.optim.Optimizer — optimizer in use.
            scheduler: torch.optim.lr_scheduler — learning rate scheduler (can be None).
            epoch: int — current epoch number.
            loss: float — current loss value.
            path: str — file path to save checkpoint.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'loss': loss,
            'loss_function_state_dict': self.loss_function.state_dict()
                if isinstance(self.loss_function, torch.nn.Module) else None,
            'ff_loss_function_state_dict': self.ff_loss_function.state_dict(),
            's11_loss_function_state_dict': self.s11_loss_function.state_dict(),
        }

        torch.save(checkpoint, path)


    def load_checkpoint(self, path="checkpoint.pth"):
        """
        Load training checkpoint including model, optimizer, and scheduler state.

        Args:
            model: torch.nn.Module — your model.
            optimizer: torch.optim.Optimizer — optimizer in use.
            scheduler: torch.optim.lr_scheduler — learning rate scheduler (can be None).
        """
        if not os.path.isfile(path):
            print(f"No checkpoint found at {path}")
            return None

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint.get('loss_function_state_dict') is not None:
            self.loss_function.load_state_dict(checkpoint['loss_function_state_dict'])
        if checkpoint.get('ff_loss_function_state_dict') is not None:
            self.ff_loss_function.load_state_dict(checkpoint['ff_loss_function_state_dict'])
        if checkpoint.get('s11_loss_function_state_dict') is not None:
            self.s11_loss_function.load_state_dict(checkpoint['s11_loss_function_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print(f"Checkpoint loaded from {path}, resuming at epoch {epoch} with loss {loss}")
        return epoch, loss
