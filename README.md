# CCNU LSKT
# Learning Process Segment-based Knowledge Tracing with Cognitive Load Theory
ICDE'2022: Learning Process Segment-based Knowledge Tracing with Cognitive Load Theory (Pytorch implementation for LSKT).




If you find this code useful in your research then please cite  
```bash
@inproceedings{bin2022LSKT,
  title={Learning Process Segment-based Knowledge Tracing with Cognitive Load Theory},
  author={Tao Huang, Bin Chen, Huali Yang, Jing Geng,Hekun Xie and Tao Yu},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2020}
}
``` 

## Setups
The requiring environment is as bellow:  

- Linux 
- Python 3+
- PyTorch 1.2.0 
- Scikit-learn 0.21.3
- Scipy 1.3.1
- Numpy 1.17.2



## Running LSKT.
Here are some examples for using LSKT model (on ASSISTments2009 datasets):
```
python main.py --dataset assist2009_updated --model lskt 
```



Contact: Bin Chen (chenbin@mails.ccnu.edu.cn).

Research in CCNU about ML, DL, Knowledge Tracing ,Knowledge Graph ,Cognitive Diagnosis and Personalized Learning

Abstract:
With the increasing demands of personalized learning, knowledge tracing has become important which traces students’ knowledge states based on their historical practices. Recent developments in KT using flexible deep neural network-based models excel at this task. These methods center around deep sequential models. However, these models don’t consider the human cognitive load. As we can know from Cognitive Load Theory (CLT), the student’s cognitive process is composed of segments of the learning process instead of strictly continuous during practice due to the working memory. In this paper, we propose Learning Process Segment-based Knowledge Tracing with Cognitive Load Theory (LSKT). LSKT uses a component named segment-based temporal attention to model working memory and long-term memory, which marry causal convolution and attention. In order to adapt to different students, different convolution kernel sizes are used to extract segment feature for the same interaction sequence. And then all of these features will be computed by Student Spatial Attention to best match the current student. We conduct experiments on several real-world benchmark datasets and show that LSKT outperforms existing KT methods on predicting future student responses, even by up to 3.4% in some case than state-of-the-art model. We also conduct several ablation studies and show that our key innovations are effective.
