import torch
from torch.utils.data import DataLoader

import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.lobes.features import MFCC, Fbank
from speechbrain.nnet.losses import nll_loss
from speechbrain.utils.checkpoints import Checkpointer

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from hyperpyyaml import load_hyperpyyaml
import os
import sys
import numpy as np
import librosa
from torchbnn.utils import freeze, unfreeze


class IntentRec(sb.Brain):
    def compute_forward(self, batch, stage):
        #"Given an input batch it computes the output probabilities."
        batch = batch.to(self.device)
        sig, lens = batch.sig
        
        outputs = self.modules.wav2vec2(sig)
        # average the layers
        #outputs = torch.mean(outputs, dim=0)
        outputs = outputs[19]

        # AdaptativeAVG pool
        outputs = self.hparams.avg_pool(outputs, lens)
        outputs = outputs.view(outputs.shape[0], -1)

        outputs_req = self.modules.bayesian_layer(outputs)
        outputs_req = self.hparams.log_softmax(outputs_req)

        return outputs_req


    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss."""
        req_lab, _ = batch.labels_encoded_request

        outputs_req = predictions
        # nll_loss = self.hparams.compute_cost(outputs_req, req_lab.squeeze(-1))
        # used when we want to additionally use class weights
        loss = ce_loss(outputs_req, req_lab.squeeze(-1))
        kl_loss = self.hparams.kl_cost()
        loss = loss + kl_loss

        if stage != sb.Stage.TRAIN:
            self.f_metrics_req.append(batch.id, outputs_req, req_lab)

            pred_req = torch.topk(outputs_req, k=1, dim=1)[1]
            recall = recall_score(req_lab.cpu().numpy(), pred_req.cpu().numpy(), average="macro")
            self.recall.append(recall)

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
        if stage != sb.Stage.TRAIN:
            self.f_metrics_req = self.hparams.f_score_error_stats()
            self.recall = []


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
            self.recall = torch.tensor(self.recall).mean()
            stats = {
                "loss": stage_loss,
                "F1": self.f_metrics_req.summarize(field="F-score"),
                "Recall": self.recall,
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
                meta=stats, max_keys=["Recall"]
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
            true_labels_req = []
            pred_labels_req = []
            true_labels_com = []
            pred_labels_com = []
            file_ids = []
            probs = []
            for batch in dataset:
                outputs_req = self.compute_forward(batch, stage=sb.Stage.TEST) 
                #outputs_req = torch.topk(outputs_req, k=1, dim=1)[1]
                outputs_req_probs, outputs_req = torch.topk(outputs_req, k=1, dim=1)
 
                topk_req = outputs_req.squeeze(0)

                req_lab, _ = batch.labels_encoded_request
                req_lab = req_lab.squeeze(0)

                topk_req = topk_req.cpu().detach().numpy()
                req_lab = req_lab.cpu().detach().numpy()

                for i in range(len(req_lab)):
                    file_ids.append(batch.id[i])
                    probs.append(torch.exp(outputs_req_probs).item())

                    if req_lab[i] in [0, 2]:
                        true_labels_req.append(0)
                    else:
                        true_labels_req.append(1)
                    if topk_req[i] in [0, 2]:
                        pred_labels_req.append(0)
                    else:
                        pred_labels_req.append(1)

                    if req_lab[i] in [0, 1]:
                        true_labels_com.append(0)
                    else:
                        true_labels_com.append(1)
                    if topk_req[i] in [0, 1]:
                        pred_labels_com.append(0)
                    else:
                        pred_labels_com.append(1)

        
        true_req = np.array(true_labels_req)
        pred_req = np.array(pred_labels_req)
        true_com = np.array(true_labels_com)
        pred_com = np.array(pred_labels_com)


        # format the printing in a pretty way
        print("Requests")
        print("Accuracy: %.4f" % accuracy_score(true_req, pred_req))
        print("UAR: %.4f" % balanced_accuracy_score(true_req, pred_req))
        print("UAR complaint: %.4f" % balanced_accuracy_score(true_com, pred_com))
        print("F1: %.4f" % f1_score(true_req, pred_req, average="macro"))


def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "train_dev.json"), replacements={"data_root": data_folder})
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "devel.json"), replacements={"data_root": data_folder})
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "devel.json"), replacements={"data_root": data_folder})
    
    datasets = [train_data, valid_data, test_data]
    
    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(file_path):
        file_id = file_path.split("/")[-1]

        # Speechbrain loader 
        sig, sr = librosa.load(file_path, sr=16000)

        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("labels_encoded_request")
    def text_pipeline(label):
        #label = "no_presta"
        labels_encoded_request = hparams["label_encoder_request"].encode_sequence_torch([label])
        yield labels_encoded_request


    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
    
    hparams["label_encoder_request"].update_from_didataset(train_data, output_key="label")

    # save the encoder
    hparams["label_encoder_request"].save(hparams["label_encoder_file_request"])
    
    # load the encoder
    hparams["label_encoder_request"].load_if_possible(hparams["label_encoder_file_request"])

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "labels_encoded_request"])
    
    train_data = train_data.filtered_sorted(sort_key="length", reverse=False)
    
    return train_data, valid_data, test_data


def main(device="cuda"):
    global ce_loss
    #class_weights = torch.tensor([0.83439335, 0.75598404, 0.75598404, 0.75598404]).to(device) # used for train
    class_weights = torch.tensor([0.84551041, 0.7450361, 1.08002617, 1.82095588]).to(device) # used for train+dev
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

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
    train_data, valid_data, test_data = data_prep("../data/speechbrain_splits_4_classes/requests", hparams)

    # Load pretrained model (this should be commented if we want to continue training because it will load the pretrained model)
    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    
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
        int_brain.run_inference(test_data, "Recall", hparams["test_dataloader_options"])


if __name__ == "__main__":
    main()
