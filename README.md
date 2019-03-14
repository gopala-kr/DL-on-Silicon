

---------

        It’s likely that, within the next two to three years, the AI industry will converge around one open-source cross-compilation supported by all front-end and back-end environments. Other industry developments in recent weeks call attention to the opening of the AI cross-compilation ecosystem.

        These include:

        - Microsoft’s open-sourcing of a GitHub repo to foster cross-framework benchmarking of GPU-optimized deep learning models.
        - ARM’s partnership with Nvidia to integrate the open-source Nvidia Deep Learning Accelerator architecture into its just-announced Project Trillium platform, designed to enable cross-framework deep learning model compilation for optimized execution in mobile, “internet of things” and other mass-market edge devices.
        - IBM’s launch of the new open-source Fabric for Deep Learning framework, which supports optimized deployment of deep-learning microservices built in TensorFlow, PyTorch, Caffe2 and other frameworks to diverse compute nodes on heterogeneous clusters of GPUs and CPUs via stateless RESTful interfaces over Kubernetes.
        - Linux Foundation’s launch of the Acumos AI Project, which defines APIs, an open-source framework, and an AI model catalog for framework-agnostic AI app development, chaining and deployment over Kubernetes.


 More details on [DL compilers](https://github.com/gopala-kr/a-week-in-wild-ai/tree/master/12-ai-hardware-compilers)


      The aim of this project is to do the fundamental research and experimentation on
         - Existing/emerging ML/DL compiler optimization technologies 
         - Different hardwares/computing platforms which can accelerate ML/DL model training and inference(active research area)
         - Research on each computing platform and code optimization/synthesis from high level to low level machine code(active research area)
         - To find out how the existing platforms differ from each other and their offerings
         - To bring out the best in all frameworks, which helps in creating single open platform/standard.
         - To initiate a single cross-compilation open platform which simplifies integration of future frontend and backend environments.
         - To publish a papers on this research
         
------------------


- [Review Papers](https://github.com/gopala-kr/DL-on-Silicon/blob/master/review_papers.md)

        

##### References

- [AI-accelerators(CPU, GPU, FPGA, ASIC, SoC, HPC), Neuromorphic and Quantum Compute for AI](https://github.com/gopala-kr/a-week-in-wild-ai/tree/master/01-ai-accelerators)
- [case study: ml/dl frameworks/libraries(an architectural overview)](https://github.com/gopala-kr/10-weeks/tree/master/Projects-Blogs/06-ml-dl-frameworks)
- [deep learning compiler Optimization for different hardwares](https://github.com/gopala-kr/a-week-in-wild-ai/tree/master/12-ai-hardware-compilers)
- [Demystifying Deep Learning Frameworks and Libraries(end to end low+high level)](https://github.com/gopala-kr/a-week-in-wild-ai/tree/master/14-demystifying-dl-frameworks-and-libraries)
- [Neural Networks on Silicon](https://github.com/fengbintu/Neural-Networks-on-Silicon)
- [Embedded-Neural-Network](https://github.com/ZhishengWang/Embedded-Neural-Network)
- [MyCaffe: A Complete C# Re-Write of Caffe with Reinforcement Learning](https://arxiv.org/ftp/arxiv/papers/1810/1810.02272.pdf)
- [Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions](https://arxiv.org/abs/1802.04730)
- [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/abs/1802.04799)
- [Toolflows for Mapping Convolutional Neural Networks on FPGAs:
A Survey and Future Directions](https://arxiv.org/pdf/1803.05900v1.pdf)
- [The implementation of a Deep Recurrent Neural
Network Language Model on a Xilinx FPGA](https://arxiv.org/ftp/arxiv/papers/1710/1710.10296.pdf)
- [Automatic Full Compilation of Julia Programs and ML Models to Cloud TPUs](https://arxiv.org/pdf/1810.09868v1.pdf)
- [Facebook: Hardware & Software Systems research](https://research.fb.com/announcing-the-winners-of-the-facebook-hardware-software-systems-research-awards/)
- [Intel AI: high-throughput-object-detection-on-edge-platforms-with-fpga](https://ai.intel.com/high-throughput-object-detection-on-edge-platforms-with-fpga/)
- [Intel AI: adaptable-deep-learning-solutions-with-ngraph-compiler-and-onnx](https://ai.intel.com/adaptable-deep-learning-solutions-with-ngraph-compiler-and-onnx/)
- [Intel AI: intel-fpgas-powering-real-time-ai-inferencing](https://ai.intel.com/intel-fpgas-powering-real-time-ai-inferencing/)
- [Intel AI: heterogenous-computing-ai-hardware-designed-for-specific-tasks ](https://ai.intel.com/heterogenous-computing-ai-hardware-designed-for-specific-tasks/)
- [Intel AI Academy: Framework Optimization Training](https://software.intel.com/en-us/ai-academy/frameworks)
- [Intel® Optimization for TensorFlow*](https://software.intel.com/en-us/videos/introduction-to-intel-optimization-for-tensorflow)
- [Intel AI Docs](https://ai.intel.com/docs/) 
- [CUDA](https://github.com/Erkaman/Awesome-CUDA) - useful libraries and resources for CUDA development
- [IBM AI: Hardware and the Physics of AI](https://www.research.ibm.com/artificial-intelligence/physics-of-ai/)
- [DyNet: The Dynamic Neural Network Toolkit](https://arxiv.org/pdf/1701.03980.pdf)
- [Deep Learning with Dynamic Computation Graphs](https://arxiv.org/pdf/1702.02181.pdf)
- [Mesh-TensorFlow: Deep Learning for Supercomputers](https://arxiv.org/pdf/1811.02084v1.pdf)
- [AI Benchmark: Running Deep Neural Networks
on Android Smartphones](https://arxiv.org/pdf/1810.01109v1.pdf)
- [Apple: Democratizing Production-Scale Distributed Deep Learning](https://arxiv.org/pdf/1811.00143v1.pdf)
- [Communication Primitives in Deep Learning Frameworks](https://medium.com/south-park-commons/communication-primitives-in-deep-learning-frameworks-7faefb2f3f63)
- [Comparing Deep Learning Frameworks: A Rosetta Stone Approach](https://blogs.technet.microsoft.com/machinelearning/2018/03/14/comparing-deep-learning-frameworks-a-rosetta-stone-approach/)
- [CodeX: Bit-Flexible Encoding for Streaming-based FPGA Acceleration of DNNs](https://arxiv.org/abs/1901.05582v1)
- [EcoRNN: Efficient Computing of LSTM RNN Training on GPUs](https://arxiv.org/abs/1805.08899v4)
- [Training for 'Unstable' CNN Accelerator:A Case Study on FPGA](https://arxiv.org/abs/1812.01689v1)
- [Optimized Compilation of Aggregated Instructions for Realistic Quantum Computers](https://arxiv.org/abs/1902.01474v1)
- [Software-Defined FPGA Accelerator Design for Mobile Deep Learning Applications](https://arxiv.org/abs/1902.03192v1)
- [Systimator: A Design Space Exploration Methodology for Systolic Array based CNNs Acceleration on the FPGA-based Edge Nodes](https://arxiv.org/abs/1901.04986v2)
- [TensorSCONE: A Secure TensorFlow Framework using Intel SGX](https://arxiv.org/abs/1902.04413v1)
- [Lingvo: a Modular and Scalable Framework for Sequence-to-Sequence Modeling](https://arxiv.org/abs/1902.08295v1)

------------


_**Maintainer**_

Gopala KR /@gopala-kr

------------
