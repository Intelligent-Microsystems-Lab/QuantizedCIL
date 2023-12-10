import torch

class Forgetting:
    def __init__(self):
        self.max_acc_cls_dic = {}
        self.forgetting_scores = [] 

    def __call__(self, acc_cls_dic):
        self.update_max_acc_cls_dic(acc_cls_dic)

        if self.forgetting_scores:
            self.forgetting_scores.append(self.forgetting(self.max_acc_cls_dic,
                                                          acc_cls_dic))
        else:
            # first step has no forgetting
            self.forgetting_scores.append(None)
        return self.forgetting_scores[-1]
    
    def update_max_acc_cls_dic(self, acc_cls_dic):
        for cls, acc in acc_cls_dic.items():
            if cls not in self.max_acc_cls_dic:
                self.max_acc_cls_dic[cls] = acc
            else:
                self.max_acc_cls_dic[cls] = max(self.max_acc_cls_dic[cls], acc)

    def forgetting(self, max_acc_cls_dic, acc_cls_dic):
        forgetting = 0
        for cls in self.max_acc_cls_dic:
            forgetting += max_acc_cls_dic[cls] - acc_cls_dic[cls]
        forgetting = forgetting / len(max_acc_cls_dic)
        return round(forgetting,2)