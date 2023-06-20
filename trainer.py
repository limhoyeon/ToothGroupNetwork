import torch
import wandb
from loss_meter import LossMeter
from math import inf
class Trainer:
    def __init__(self, config = None, model=None, gen_set=None):
        self.gen_set = gen_set
        self.config = config
        self.model = model

        self.val_count = 0
        self.train_count = 0
        self.step_count = 0
        if config["wandb"]["wandb_on"]:
            wandb.init(
            entity=self.config["wandb"]["entity"],
            project=self.config["wandb"]["project"],
            notes=self.config["wandb"]["notes"],
            tags=self.config["wandb"]["tags"],
            name=self.config["wandb"]["name"],
            config=self.config,
            )

        self.best_val_loss = inf

    def train(self, epoch, data_loader):
        total_loss_meter = LossMeter()
        step_loss_meter =  LossMeter()
        pre_step = self.step_count
        for batch_idx, batch_item in enumerate(data_loader):
            loss = self.model.step(batch_idx, batch_item, "train")
            torch.cuda.empty_cache()
            total_loss_meter.aggr(loss.get_loss_dict_for_print("train"))
            step_loss_meter.aggr(loss.get_loss_dict_for_print("step"))
            print(loss.get_loss_dict_for_print("step"))
            if ((batch_idx+1) % self.config["tr_set"]["scheduler"]["schedueler_step"] == 0) or (self.step_count == pre_step and batch_idx == len(data_loader)-1):
                if self.config["wandb"]["wandb_on"]:
                    wandb.log(step_loss_meter.get_avg_results(), step=self.step_count)
                    wandb.log({"step_lr": self.model.scheduler.get_last_lr()[0]}, step = self.step_count)
                self.step_count +=1
                self.model.scheduler.step(self.step_count)
                step_loss_meter.init()
                
        if self.config["wandb"]["wandb_on"]:
            wandb.log(total_loss_meter.get_avg_results(), step = self.step_count)
            self.train_count += 1
        self.model.save("train")

    def test(self, epoch, data_loader, save_best_model):
        total_loss_meter = LossMeter()
        for batch_idx, batch_item in enumerate(data_loader):
            loss = self.model.step(batch_idx, batch_item, "test")
            total_loss_meter.aggr(loss.get_loss_dict_for_print("val"))

        avg_total_loss = total_loss_meter.get_avg_results()
        if self.config["wandb"]["wandb_on"]:
            wandb.log(avg_total_loss, step = self.step_count)
            self.val_count+=1

        if save_best_model:
            if self.best_val_loss > avg_total_loss["total_val"]:
                self.best_val_loss = avg_total_loss["total_val"]
                self.model.save("val")

    def train_depr(self):
        total_loss = 0
        step_loss = 0
        for batch_idx, batch_item in enumerate(self.train_loader):
            loss = self.model.step(batch_idx, batch_item, "train")
            total_loss += loss
            step_loss += loss
            if (batch_idx+1) % self.config["tr_set"]["schedueler_step"] == 0:
                self.model.scheduler.step()
                step_loss /= self.config["tr_set"]["schedueler_step"]
                if self.config["wandb"]["wandb_on"]:
                    wandb.log({"step_train_loss":step_loss})
                step_loss = 0
        total_loss /= len(self.train_loader)
        if self.config["wandb"]["wandb_on"]:
            wandb.log({"train_loss": total_loss})
        self.model.save("train")

    def test_depr(self):
        total_loss = 0
        for batch_idx, batch_item in enumerate(self.val_loader):
            loss = self.model.step(batch_idx, batch_item, "test")
            total_loss += loss
        total_loss /= len(self.val_loader)
        if self.config["wandb"]["wandb_on"]:
            wandb.log({"val_loss": total_loss})

        if self.best_val_loss > total_loss:
            self.best_val_loss = total_loss
            self.model.save("val")
    
    def run(self):
        train_data_loader = self.gen_set[0][0]
        val_data_loader = self.gen_set[0][1]
        for epoch in range(100000):
            self.train(epoch, train_data_loader)
            self.test(epoch, val_data_loader, True)