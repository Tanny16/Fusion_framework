import os
import copy
import torch

from trainer.fusion_trainer import train_process, val_process, predict
from utils.data_pipline import StockNewsDataSet
from model.model_baseline import BaselineModel
from utils.preprocess import BertVec
from utils.Log_process import Logger


# --------------------global parameters-------------------

work_dir = "./workdir/baseline/"
train_data_path = "data/data.train"
val_data_path = "data/data.dev"
test_data_path = "data/data.test"

bv = BertVec()
num_epochs = 10
num_sentences = 10
num_words = 15
num_chars = 4
vec_length = 768
news_vec_size = 32
num_base = 8

num_classes = 2
labels = ["up", "down"]
lr_init = 0.01
GPU_id = "cuda:0"

os.makedirs(work_dir, exist_ok=True)
device = torch.device(GPU_id if torch.cuda.is_available() else "cpu")
logger = Logger(file_path="{}log.log".format(work_dir))


# --------------------data pipeline-------------------

datasets_train = StockNewsDataSet(train_data_path,  bv, vec_length, num_sentences, num_words, num_chars)
datasets_val = StockNewsDataSet(val_data_path,  bv, vec_length, num_sentences, num_words, num_chars)



# If num_workers is not 0, it will deadlock (see BaselineDataSet for the reason)

dataLoader_train = torch.utils.data.DataLoader(datasets_train,
                                               batch_size=32,
                                               shuffle=True,
                                               num_workers=0)

dataLoader_val = torch.utils.data.DataLoader(datasets_val,
                                             batch_size=32,
                                             shuffle=False,
                                             num_workers=0)


# --------------------model-------------------

model = BaselineModel(input_size=vec_length, num_classes=num_classes, news_vec_size=news_vec_size, num_base=num_base)
model = model.to(device)
model.train(mode=True)

# --------------------loss function and optimizer-------------------

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.99))
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


# --------------------params-------------------

params = {
    "model": model,
    "optimizer": optimizer,
    "criterion": criterion,
    "device": device,
    "bv": bv,
    "vec_length": vec_length,
    "logger": logger,
    "num_sentences": num_sentences,
    "num_words": num_words,
    "num_chars": num_chars
}

# --------------------train and val-------------------

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

print("Start training:")
print("Detailed training log path: {}".format("{}log.log".format(work_dir)))

for epoch in range(num_epochs):
    epoch_index = "{}/{}".format(epoch + 1, num_epochs)
    train_process(dataLoader_train, epoch_index, params)
    val_acc = val_process(dataLoader_val, epoch_index, params)
    exp_lr_scheduler.step()
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
    torch.save(model.state_dict(), "{}epoch-{}.pkl".format(work_dir, epoch))

model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), "{}best-val.pkl".format(work_dir))

# --------------------inference-------------------

# model.load_state_dict(torch.load("{}epoch-0.pkl".format(work_dir), map_location=lambda storage, loc: storage))
# model.eval()
#
# # The following parameters can be reconfigured, and they can be inconsistent with training.
# # Training is because of Batch training so it needs to be consistent.
# params.update({
#     "num_sentences": 12,
#     "num_words": 50,
#     "num_chars": 5
# })
#
# print("Start testing:")
# predict(test_data_path, work_dir, params)