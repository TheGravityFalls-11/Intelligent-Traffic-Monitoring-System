import os

train_path = "dataset/train/images"
valid_path = "dataset/valid/images"

with open("dataset/train.txt","w") as f:
    for img in os.listdir(train_path):
        f.write(train_path + "/" + img + "\n")

with open("dataset/valid.txt","w") as f:
    for img in os.listdir(valid_path):
        f.write(valid_path + "/" + img + "\n")

print("TXT files created")