from src.losses.losses import FarfeildLoss, S11Loss, MSSSIMLoss, MSELoss, compute_snr, compute_metrics
import torch
from tqdm import tqdm
import wandb
import random
import os
from src.utils import scale_to_log
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
        self.s11_loss_function = S11Loss(self.config)

        self.learning_rate = self.config.get('learning_rate')
        self.num_epochs = 500

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
        model_weights_path = cwd + '/trained_models/' + self.config.get('path_to_save_model_weights')

        for epoch in tqdm(range(self.num_epochs), desc="Training GNN ", unit="epoch", leave=True):
            train_losses = []
            train_accuracy = []
            train_accuracy_0_4 = []
            test_loss = 0

            # train the network:
            self.model.train()
            for idx, (graph, s11, ff_images, graph_identifier, params_vector, raw_idx) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Examples in training", unit=' examples', leave=True):
                predictions, loss = self.train_step([params_vector, s11, ff_images, raw_idx])
                train_losses.append(loss)

            self.scheduler.step()
            train_loss = sum(train_losses)/len(train_losses)

            # test the network:
            self.model.eval()
            test_loss = self.test_network(test_loader=test_loader)

            # Check if the current validation loss is the best so far
            if test_loss < best_test_loss:
                best_test_loss = test_loss

                # Save the model's state_dict to the file
                torch.save(self.model.state_dict(), model_weights_path)
                print(f'Epoch {epoch}: Validation current best Loss: {test_loss}')

            self.validate(test_loader=test_loader)
            self.wandb_log_dict.update({
                "GNN Train Loss:": train_loss,
                "GNN Test Loss:": test_loss,
                'Learning rate': self.optimizer.param_groups[0]['lr'],
            })
            self.wandb_run.log(self.wandb_log_dict)


    def train_step(self, batch):
        params_vector, s11, ff_images, raw_idx = batch
        if len(params_vector['ant_parameters']) == 1:
            matrix = params_vector['ant_parameters'].unsqueeze(0)
        else:
            matrix = params_vector['ant_parameters'][0].unsqueeze(1)

        if self.config.get('data_set_name') == 'FMNIST_CIFAR':
            reflector_matrix = params_vector['reflector_matrix'].unsqueeze(1)
            matrix = matrix
            matrix = torch.cat([matrix, reflector_matrix], dim=1)

        self.optimizer.zero_grad()

        loss_list = []
        env_vector = self.env_paramters_to_vector(params_vector['env_parameters'], params_vector['reflectors_params'])
        pred = self.model(matrix.to(self.device))
        ff_loss = self.ff_loss_function(pred, ff_images[:, self.config.get('idx_freq'), :, :].unsqueeze(1).to(self.device), train=True)
        loss_list.append(ff_loss.item())
        ff_loss.backward()
        self.optimizer.step()

        loss = ff_loss.item()
        predictions = {}
        loss_metrics = loss
        return predictions, loss_metrics


    def test_network(self, test_loader=None):

        test_losses = []
        total_metrics = {'MAE': 0.0, 'MSE': 0.0, 'max_error': 0.0, 'SNR': 0.0, 'mssim': 0.0}
        num_of_abslute_examples = 0
        self.model.eval()
        for idx, (graph, s11, ff_images, graph_identifier, example_paramters, raw_idx) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Examples in testing", unit=' examples', leave=True):

            example_loss = 0

            if len(example_paramters['ant_parameters']) == 1:
                matrix = example_paramters['ant_parameters'].unsqueeze(0)
            else:
                matrix = example_paramters['ant_parameters'][0].unsqueeze(1)

            if self.config.get('data_set_name') == 'FMNIST_CIFAR':
                reflector_matrix = example_paramters['reflector_matrix'].unsqueeze(1)
                matrix = matrix
                matrix = torch.cat([matrix, reflector_matrix], dim=1)


            with torch.no_grad():
                env_vector = self.env_paramters_to_vector(example_paramters['env_parameters'], example_paramters['reflectors_params'], is_training=False)
                pred = self.model(matrix.to(self.device))
                ff_loss = self.ff_loss_function(pred, ff_images[:, self.config.get('idx_freq'), :, :].unsqueeze(1).to(self.device), train=False)

                example_metrics = compute_metrics(pred, ff_images[:, self.config.get('idx_freq'), :, :].unsqueeze(1).to(self.device), self.ff_stats)
                # Accumulate metrics
                for key, value in example_metrics.items():
                    total_metrics[key] += value
                num_of_abslute_examples += 1

            batch_loss = ff_loss
            test_losses.append(batch_loss.item())


        test_metrics = {key: value / num_of_abslute_examples for key, value in total_metrics.items()}
        self.wandb_log_dict.update(test_metrics)
        print('test_metrics: ', test_metrics)
        test_loss = sum(test_losses)/len(test_losses)
        loss_hist_image = self.loss_list_to_hist_image(test_losses)
        self.wandb_log_dict.update({"farafeild loss histogram dB-dB:": loss_hist_image})

        return test_loss


    def compute_metrics(self, prediction: torch.Tensor, ground_truth: torch.Tensor):
        """calculats the metrics between the pridictionn and the ground truth
        contains MAE,MSE, max avarge error, snr, mssim, and more to come...

        Args:
            prediction (torch.Tensor): ff_pred in db
            ground_truth (torch.Tensor):ff_target in db
        """
        abs_error = abs(ground_truth-prediction)
        mae = abs_error.mean()
        mse = MSELoss()(prediction, ground_truth)
        max_error = abs_error.max()
        snr = compute_snr(prediction.squeeze(-1), ground_truth.squeeze(-1))
        mssim = 1 - MSSSIMLoss()(prediction, ground_truth, self.ff_stats)
        return {'MAE': mae, 'MSE': mse, 'max_error': max_error, 'SNR': snr, 'mssim': mssim}


    def init_wandb(self):
        # start a new wandb run to track this script
        self.wandb_run = wandb.init(
            # set the wandb project where this run will be logged
            project="antenna GNN farfeild directivty",
            # track hyperparameters and run metadata
            config={
                "learning_rate": self.config.get('learning_rate'),
                "architecture": "antenna GNN farfeild directivty",
                "dataset": "Alpha datset",
                "epochs": self.config.get('num_epochs')
            }
        )
        self.wandb_log_dict = {}


    def validate(self, test_loader):
        # Convert loader to a list to grab random samples
        data_list = list(test_loader)
        # Grab random examples
        random.seed(0)
        random_samples = random.sample(data_list, self.config.get('number_of_validation_samples'))
        for example_number, (graph, s11, ff_images, graph_identifier, example_paramters, raw_idx) in enumerate(random_samples):

            if len(example_paramters['ant_parameters']) == 1:
                matrix = example_paramters['ant_parameters'].unsqueeze(0)
            else:
                matrix = example_paramters['ant_parameters'][0].unsqueeze(0)
            if self.config.get('data_set_name') == 'FMNIST_CIFAR':
                reflector_matrix = example_paramters['reflector_matrix'].unsqueeze(1)
                matrix = torch.cat([matrix, reflector_matrix], dim=1)

            image_list = []
            with torch.no_grad():
                raw_idx = str(raw_idx)
                example_loss = 0
                env_vector = self.env_paramters_to_vector(example_paramters['env_parameters'], example_paramters['reflectors_params'], is_training=False)
                pred = self.model(matrix.to(self.device))
                ff_label = ff_images[:, self.config['idx_freq'], :, :].unsqueeze(1).to(self.device)
                ff_loss = self.ff_loss_function(pred, ff_label, train=False)

                wandb_pred_image = self.tensor_to_image(pred, raw_idx, self.frequencies[self.config['idx_freq']], str_type='pred dB', mse_loss=ff_loss.item())
                image_list.append(wandb_pred_image)
                wandb_target_image = self.tensor_to_image(ff_label, raw_idx, self.frequencies[self.config['idx_freq']], str_type='target dB')
                image_list.append(wandb_target_image)

                s11_magnitude_label = s11['s11_abs']
                s11_magnitude_label = scale_to_log(s11_magnitude_label)
                s11_frequancies = s11['frequancy']
                wandb_s11_magnitude = self.s11_abs_to_image(s11_magnitude_label, s11_frequancies, raw_idx, 'dB')
                image_list.append(wandb_s11_magnitude)

                ff_loss = example_loss/len(self.frequencies)
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

    def tensor_to_image(self, tensor_image, raw_idx, frequency, str_type='loss', mse_loss=None):
        plt.figure(figsize=(25, 25))
        tensor = tensor_image[0].clone().detach().cpu() # 1, 34, 34
        tensor = tensor.squeeze(0) # 34, 34
        plt.imshow(tensor, cmap='jet', vmin=max(self.ff_stats['min'], -15), vmax=self.ff_stats['max'] + 0.1)
        plt.colorbar()
        plt.title('Tensor Visualization')
        caption = "farafeild' " + raw_idx + ":\n" + 'frequency = ' + str(frequency) + ' ' + str_type
        if mse_loss:
            caption = "farafeild' " + raw_idx + ":\n" + 'frequency = ' + str(frequency) + ' ' + str_type + "\n" + 'MSE:' + f"{mse_loss:.4f}"
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

    def env_paramters_to_vector(self, env_parameters, reflector_parameters, is_training=True):
        """Convert environment parameters to a PyTorch vector, excluding string values.
        Handles cases where reflector_parameters['thetas'] and ['phis'] can be lists or tensors."""
        def to_tensor_and_flatten(x):
            if isinstance(x, list):
                x = torch.tensor(x)
            return x.float().flatten()

        thetas = to_tensor_and_flatten(reflector_parameters['thetas'])
        phis = to_tensor_and_flatten(reflector_parameters['phis'])

        env_vector = torch.stack([
            env_parameters['h'].float(),
            env_parameters['patch_x'].float(),
            env_parameters['patch_y'].float(),
            env_parameters['ground_x'].float(),
            env_parameters['ground_y'].float(),
            env_parameters['box_size'].float(),
            env_parameters['radius'].float(),
            thetas,
            phis
        ], dim=-1)

        return env_vector
