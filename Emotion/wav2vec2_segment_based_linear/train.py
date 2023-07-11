import torch
from torch.utils.data import DataLoader

import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.lobes.features import MFCC, Fbank
from speechbrain.nnet.losses import nll_loss
from speechbrain.utils.checkpoints import Checkpointer

import torchaudio
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from hyperpyyaml import load_hyperpyyaml
import os
import sys
import statistics
import numpy as np
import librosa
from scipy import stats


class IntentRec(sb.Brain):
    def compute_forward(self, batch, stage):
        #"Given an input batch it computes the output probabilities."
        batch = batch.to(self.device)
        sig, lens = batch.sig
        
        outputs = self.modules.wav2vec2(sig)
        # average the layers
        # outputs = torch.mean(outputs, dim=0)
        # take a specific layer (layer 18 the best or the last one for the model from Yaroslav)
        outputs = outputs[18]
        # outputs = outputs[-1]

        # split the audio in overlapping windows
        output_splits = outputs.unfold(dimension=1, size=self.hparams.split_size, step=self.hparams.step_size)
        split_probs = torch.zeros((output_splits.size(0), output_splits.size(1), self.hparams.n_classes)).to(self.device)
        for i in range(output_splits.size(1)):
            split = output_splits[:, i, :, :]
            split = split.permute(0, 2, 1)
            split = self.hparams.avg_pool(split)
            split = split.squeeze(1)

            split = self.modules.out_request(split)
            split = torch.nn.functional.sigmoid(split)

            split_probs[:, i, :] = split

        # average the probabilities
        split_probs = torch.mean(split_probs, dim=1)

        return split_probs


    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss."""
        true_lab = batch.label
        true_lab = torch.tensor(true_lab, dtype=torch.float32).to(predictions.device)
        split_probs = predictions

        loss = self.hparams.compute_cost(split_probs, true_lab)

        if stage != sb.Stage.TRAIN:
            spearman_cor, spearman_p = stats.spearmanr(split_probs.cpu().flatten(), true_lab.cpu().flatten())
            self.spearman.append(spearman_cor)

        return loss

        
    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up evaluation-only statistics trackers
        #if stage != sb.Stage.TRAIN:
        #    self.f_metrics_req = self.hparams.f_score_error_stats()
        #    self.recall = []
        if stage != sb.Stage.TRAIN:
            self.spearman = []


    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            spearman = statistics.mean(self.spearman)
            stats = {
                "loss": stage_loss,
                "Spearman": spearman,
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                # {"Epoch": epoch, "lr": old_lr, "wave2vec_lr": old_lr_wav2vec2},
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=stats, max_keys=["Spearman"]
            )
     
        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start(max_key=max_key, min_key=min_key)
        
        ckpts = self.checkpointer.find_checkpoints(
                max_key=max_key,
                min_key=min_key,
        )
        model_state_dict = sb.utils.checkpoints.average_checkpoints(
                ckpts, "model" 
        )
        self.hparams.model.load_state_dict(model_state_dict)


    def run_inference(
            self,
            dataset, # Must be obtained from the dataio_function
            max_key, # We load the model with the lowest error rate
            loader_kwargs, # opts for the dataloading
        ):

        # If dataset isn't a Dataloader, we create it. 
        if not isinstance(dataset, DataLoader):
            loader_kwargs["ckpt_prefix"] = None
            dataset = self.make_dataloader(
                dataset, sb.Stage.TEST, **loader_kwargs
            )

        self.checkpointer.recover_if_possible(max_key=max_key)
        self.modules.eval() # We set the model to eval mode (remove dropout etc)
        
        # Now we iterate over the dataset and we simply compute_forward and decode
        with torch.no_grad():
            spearman_correlations = []
            pearson_correlations = []
            outputs = []
            file_ids = []
            for batch in dataset:
                outputs_req = self.compute_forward(batch, stage=sb.Stage.TEST)
                true_lab  = batch.label

                # used when saving the predictions
                outputs.append(outputs_req.squeeze(0).cpu().numpy())
                file_ids.extend(batch.id)

                for i in range(len(true_lab)):
                    corr, p_value = stats.spearmanr(outputs_req[i].cpu(), true_lab[i])
                    spearman_correlations.append(corr)

                    corr, p_value = stats.pearsonr(outputs_req[i].cpu(), true_lab[i])
                    pearson_correlations.append(corr)
            
            spearman_correlations = statistics.mean(spearman_correlations)
            pearson_correlations = statistics.mean(pearson_correlations)

            print("Spearman correlation: %.4f" % spearman_correlations)
            print("Pearson correlation: %.4f" % pearson_correlations)

            # # save the predictions in a csv file
            # with open("../submission_ready_predictions/sub_3_segment_based_test.csv", "w") as f:
            #     for i in range(len(outputs)):
            #         f.write(file_ids[i] + "," + str(outputs[i][0]) + "," + str(outputs[i][1]) + "," + str(outputs[i][2])
            #                  + "," + str(outputs[i][3]) + "," + str(outputs[i][4]) + "," + str(outputs[i][5])
            #                    + "," + str(outputs[i][6]) + "," + str(outputs[i][7]) + "," + str(outputs[i][8]) + "\n")


def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "train_dev.json"), replacements={"data_root": data_folder})
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "devel.json"), replacements={"data_root": data_folder})
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "test.json"), replacements={"data_root": data_folder})
    
    datasets = [train_data, valid_data, test_data]
    
    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(file_path):
        file_id = file_path.split("/")[-1]
        #file_path = os.path.join("/m/triton/scratch/elec/t405-puhe/p/porjazd1/ComParE2023/Requests/data/wav", file_id)

        # Speechbrain loader 
        sig, sr = librosa.load(file_path, sr=16000)

        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("label")
    def text_pipeline(label):
        #label = torch.tensor(label, dtype=torch.float32)
        yield label


    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
    
    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "label"])
    
    train_data = train_data.filtered_sorted(sort_key="length", reverse=False)
    
    return train_data, valid_data, test_data


def main(device="cuda"):
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    sb.utils.distributed.ddp_init_group(run_opts) 
    
    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    
    # Trainer initialization
    int_brain = IntentRec(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        )

 
    # Dataset creation
    train_data, valid_data, test_data = data_prep("../data/speechbrain_splits", hparams)
    

    for param in hparams["modules"]["out_request"].parameters():
        print(param.mean().item())
        print(param.std().item())


    # Training/validation loop
    if hparams["skip_training"] == False:
        print("Training...")
        ###int_brain.checkpointer.delete_checkpoints(num_to_keep=0)
        int_brain.fit(
            int_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )
    else:
        # evaluate
        print("Evaluating")
        int_brain.run_inference(test_data, "Spearman", hparams["test_dataloader_options"])


if __name__ == "__main__":
    main()
