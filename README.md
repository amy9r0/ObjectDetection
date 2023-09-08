This project is designed to detect the following categories:
0:background
1:fox
2:bear
3:hog
4:dog
5:person

These images were sourced from both COCO and ImageNet datasets and totaling ~1000. Categories were balanced to reduce bias. The annotation formats were converted to PASCAL VOC. 

Transfer learning was used on two different backbones: MobileNet, ResNet50. Average precision and mean average precision was calculated for each model variation.
