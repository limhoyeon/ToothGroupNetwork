import gen_utils as gu
class LossMeter:
    def __init__(self):
        self.loss_meter_dict = {

        }
        self.step_num = 0

    def aggr(self, loss_map: dict):
        for key in loss_map.keys():
            if key not in self.loss_meter_dict:
                self.loss_meter_dict[key] = 0
            self.loss_meter_dict[key] += loss_map[key]
        self.step_num += 1

    def get_avg_results(self):
        avg_loss_meter_dict = {}
        for key in self.loss_meter_dict.keys():
            avg_loss_meter_dict[key] = self.loss_meter_dict[key] / self.step_num
        return avg_loss_meter_dict

    def init(self):
        self.step_num = 0
        self.loss_meter_dict = {}

class LossMap:
    def __init__(self):
        self.loss_dict = {
        }

    def add_loss(self, name: str, value, weight: float):
        self.loss_dict[name] = (value, weight)
    
    def add_loss_by_dict(self: object, loss_dict: dict):
        for key in loss_dict.keys():
            if key in self.loss_dict.keys():
                raise
            self.add_loss(key, loss_dict[key][0], loss_dict[key][1])

    def del_loss(self: object, name: str):
        del self.loss_dict[name]
    
    def get_sum(self):
        summation = 0
        for key in self.loss_dict.keys():
            summation += self.loss_dict[key][0] * self.loss_dict[key][1]
        return summation
    
    def get_loss_dict_for_print(self, post_fix):
        loss_dict_for_print = {}

        for key in self.loss_dict.keys():
            loss_dict_for_print[key + "_" + post_fix] = self.loss_dict[key][0].item() * self.loss_dict[key][1]
        
        total = 0
        for key in loss_dict_for_print.keys():
            total += loss_dict_for_print[key]
        
        loss_dict_for_print["total" + "_" + post_fix] = total

        return loss_dict_for_print