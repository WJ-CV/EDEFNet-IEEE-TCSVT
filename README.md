# EDEFNet
	
Explicitly Disentangling and Exclusively Fusing for Semi-supervised Bi-modal Salient Object Detection
---
This paper has been online published by IEEE Transactions on Circuits and Systems for Video Technology.

[Paper link](https://ieeexplore.ieee.org/abstract/document/10788520)  DOI: 10.1109/TCSVT.2024.3514897

Abstract
---
Bi-modal (RGB-T and RGB-D) salient object detection (SOD) aims to enhance detection performance by leveraging the complementary information between modalities. While significant progress has been made, two major limitations persist. Firstly, mainstream fully supervised methods come with a substantial burden of manual annotation, while weakly supervised or unsupervised methods struggle to achieve satisfactory performance. Secondly, the indiscriminate modeling of local detailed information (object edge) and global contextual information (object body) often results in predicted objects with incomplete edges or inconsistent internal representations. In this work, we propose a novel paradigm to effectively alleviate the above limitations. Specifically, we first enhance the consistency regularization strategy to build a basic semi-supervised architecture for the bi-modal SOD task, which ensures that the model can benefit from massive unlabeled samples while effectively alleviating the annotation burden. Secondly, to ensure detection performance (i.e., complete edges and consistent bodies), we disentangle the SOD task into two parallel sub-tasks: edge integrity fusion prediction and body consistency fusion prediction. Achieving these tasks involves two key steps: 1) the explicitly disentangling scheme decouples salient object features into edge and body features, and 2) the exclusively fusing scheme performs exclusive integrity or consistency fusion for each of them. Eventually, our approach demonstrates significant competitiveness compared to 26 fully supervised methods, while effectively alleviating 90% of the annotation burden. Furthermore, it holds a substantial advantage over 15 non-fully supervised methods.

Network Architecture
---
![image](https://github.com/user-attachments/assets/7dda549a-5f7d-4b6f-819e-168412a6302a)

Edge Refinement and Body Disentanglement (ERBD) and Body Aggregation and Edge Disentanglement (BAED)
---
![image](https://github.com/user-attachments/assets/04fb62d1-c717-41a1-bca1-b84cce202aa0)

Edge Integrity Fusion (EIF) and Body Consistency Fusion (BCF)
---
![image](https://github.com/user-attachments/assets/e71e856d-cfff-4167-9194-2b70039e6a1d)

Pre-trained model and saliency map
---
We provide [Saliency maps, model](https://pan.baidu.com/s/1r3ERnKRbT_xfVs4eEw7iag)(code: 0825)  of our EDEFNet on RGB-D SOD datasets.

Quantitative results
---
![image](https://github.com/user-attachments/assets/967e1915-7011-4065-b397-d24d13d6b803)

Citation
===
```
@article{wang2024explicitly,
  title={Explicitly Disentangling and Exclusively Fusing for Semi-supervised Bi-modal Salient Object Detection},
  author={Wang, Jie and Kong, Xiangji and Yu, Nana and Zhang, Zihao and Han, Yahong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}
```
