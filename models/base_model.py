from abc import *
from this import d
import torch
from torch.optim.lr_scheduler import ExponentialLR
import os

class BaseModel(metaclass=ABCMeta):
    def __init__(self, config, model):
        self.config = config

        self.model = model(config)
        self.model.train()
        self.model.cuda()

        if self.config["tr_set"]["optimizer"]["NAME"] == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config["tr_set"]["optimizer"]["lr"], momentum=self.config["tr_set"]["optimizer"]["momentum"], weight_decay=self.config["tr_set"]["optimizer"]["weight_decay"])

        if self.config["tr_set"]["scheduler"]["sched"] == "exp":
            self.scheduler = ExponentialLR(self.optimizer, self.config["tr_set"]["scheduler"]["step_decay"])

    def _set_model(self, phase):
        if phase=="train":
            self.model.train()
        elif phase=="val" or phase=="test":
            pass

    def load(self):
        self.model.load_state_dict(torch.load(self.config["tr_set"]["checkpoint_path"]+".h5"))

    def save(self, phase):
        if not os.path.exists(os.path.dirname(self.config["tr_set"]["checkpoint_path"])):
            os.makedirs(os.path.dirname(self.config["tr_set"]["checkpoint_path"]), exist_ok=True)
        
        if phase=="train":
            torch.save(self.model.state_dict(), self.config["tr_set"]["checkpoint_path"]+".h5")
        elif phase=="val":
            torch.save(self.model.state_dict(), self.config["tr_set"]["checkpoint_path"]+"_val.h5")
        else:
            raise "phase is something unknown"

    @abstractmethod
    def get_loss(self):
        pass
    
    @abstractmethod
    def step(self, batch_idx, batch_item):
        pass
    
    #@abstractmethod
    def infer(self, batch_idx, batch_item):
        pass
    