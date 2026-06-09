# References and Bibliography

## Foundational Papers

### The Original Transformer
1. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).**  
   *Attention Is All You Need.*  
   NeurIPS 2017.  
   https://arxiv.org/abs/1706.03762

### LSTM and Gating
2. **Hochreiter, S., & Schmidhuber, J. (1997).**  
   *Long Short-Term Memory.*  
   Neural Computation, 9(8), 1735–1780.

3. **Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014).**  
   *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.*  
   EMNLP 2014.  
   https://arxiv.org/abs/1406.1078

4. **Pascanu, R., Mikolov, T., & Bengio, Y. (2013).**  
   *On the difficulty of training recurrent neural networks.*  
   ICML 2013.  
   https://arxiv.org/abs/1211.5063

### Attention Before Transformers
5. **Bahdanau, D., Cho, K., & Bengio, Y. (2015).**  
   *Neural Machine Translation by Jointly Learning to Align and Translate.*  
   ICLR 2015.  
   https://arxiv.org/abs/1409.0473

### Recurrent Networks
6. **Elman, J. L. (1990).**  
   *Finding Structure in Time.*  
   Cognitive Science, 14(2), 179–211.

---

## Encoder-Only Models

7. **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019).**  
   *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*  
   NAACL 2019.  
   https://arxiv.org/abs/1810.04805

8. **Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019).**  
   *RoBERTa: A Robustly Optimized BERT Pretraining Approach.*  
   https://arxiv.org/abs/1907.11692

9. **Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019).**  
   *DistilBERT, a distilled version of BERT.*  
   https://arxiv.org/abs/1910.01108

---

## Decoder-Only Models (LLMs)

10. **Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019).**  
    *Language Models are Unsupervised Multitask Learners (GPT-2).*  
    OpenAI Blog.  
    https://openai.com/research/language-unsupervised

11. **Brown, T., Mann, B., Ryder, N., ... & Amodei, D. (2020).**  
    *Language Models are Few-Shot Learners (GPT-3).*  
    NeurIPS 2020.  
    https://arxiv.org/abs/2005.14165

12. **Touvron, H., Lavril, T., Izacard, G., ... & Lample, G. (2023).**  
    *LLaMA: Open and Efficient Foundation Language Models.*  
    https://arxiv.org/abs/2302.13971

13. **Touvron, H., Martin, L., Stone, K., ... & Scialom, T. (2023).**  
    *Llama 2: Open Foundation and Fine-Tuned Chat Models.*  
    https://arxiv.org/abs/2307.09288

14. **Jiang, A. Q., Sablayrolles, A., Mensch, A., ... & Roux, G. (2023).**  
    *Mistral 7B.*  
    https://arxiv.org/abs/2310.06825

---

## Encoder-Decoder Models

15. **Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020).**  
    *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5).*  
    JMLR 2020.  
    https://arxiv.org/abs/1910.10683

16. **Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2020).**  
    *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.*  
    ACL 2020.  
    https://arxiv.org/abs/1910.13461

---

## Training Techniques

17. **Kingma, D. P., & Ba, J. (2015).**  
    *Adam: A Method for Stochastic Optimization.*  
    ICLR 2015.  
    https://arxiv.org/abs/1412.6980

18. **Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).**  
    *Rethinking the Inception Architecture for Computer Vision.* (Label smoothing)  
    CVPR 2016.

19. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).**  
    *Deep Residual Learning for Image Recognition.* (Residual connections)  
    CVPR 2016.  
    https://arxiv.org/abs/1512.03385

20. **Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016).**  
    *Layer Normalization.*  
    https://arxiv.org/abs/1607.06450

---

## Positional Encoding Variants

21. **Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).**  
    *RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE).*  
    https://arxiv.org/abs/2104.09864

22. **Press, O., Smith, N. A., & Lewis, M. (2022).**  
    *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (ALiBi).*  
    ICLR 2022.  
    https://arxiv.org/abs/2108.12409

---

## Efficient Transformers

23. **Beltagy, I., Peters, M. E., & Cohan, A. (2020).**  
    *Longformer: The Long-Document Transformer.*  
    https://arxiv.org/abs/2004.05150

24. **Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022).**  
    *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.*  
    NeurIPS 2022.  
    https://arxiv.org/abs/2205.14135

25. **Dao, T. (2023).**  
    *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.*  
    https://arxiv.org/abs/2307.08691

---

## Vision Transformers

26. **Dosovitskiy, A., Beyer, L., Kolesnikov, A., ... & Houlsby, N. (2021).**  
    *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT).*  
    ICLR 2021.  
    https://arxiv.org/abs/2010.11929

27. **Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021).**  
    *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.*  
    ICCV 2021.  
    https://arxiv.org/abs/2103.14030

---

## Science Applications

28. **Jumper, J., Evans, R., Pritzel, A., ... & Hassabis, D. (2021).**  
    *Highly accurate protein structure prediction with AlphaFold.*  
    Nature.  
    https://doi.org/10.1038/s41586-021-03819-2

---

## Foundation Models and Alignment

29. **Bommasani, R., Hudson, D. A., Aditi, E., ... & Liang, P. (2021).**  
    *On the Opportunities and Risks of Foundation Models.*  
    Stanford CRFM.  
    https://arxiv.org/abs/2108.07258

30. **Ouyang, L., Wu, J., Jiang, X., ... & Lowe, R. (2022).**  
    *Training language models to follow instructions with human feedback (InstructGPT).*  
    NeurIPS 2022.  
    https://arxiv.org/abs/2203.02155

31. **Bai, Y., Jones, A., Ndousse, K., ... & Kaplan, J. (2022).**  
    *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.*  
    https://arxiv.org/abs/2204.05862

---

## Textbooks and Courses

32. **Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023).**  
    *Dive into Deep Learning.*  
    https://d2l.ai

33. **Rothman, D. (2022).**  
    *Transformers for Natural Language Processing, 2nd Edition.*  
    Packt Publishing.

34. **Jurafsky, D., & Martin, J. H. (2023).**  
    *Speech and Language Processing, 3rd Edition.*  
    https://web.stanford.edu/~jurafsky/slp3/

35. **Stanford CS224N: Natural Language Processing with Deep Learning (2024).**  
    https://web.stanford.edu/class/cs224n/

---

## Code and Implementations

36. **Karpathy, A. (2025).**  
    *microgpt.py — minimal GPT implementation.*  
    https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

37. **Wolf, T., Debut, L., Sanh, V., ... & Rush, A. (2020).**  
    *Transformers: State-of-the-Art Natural Language Processing.*  
    EMNLP 2020 (HuggingFace library).  
    https://github.com/huggingface/transformers

38. **Olah, C. (2015).**  
    *Understanding LSTM Networks.*  
    https://colah.github.io/posts/2015-08-Understanding-LSTMs/
