# Script Thuyết Trình: REPA - Representation Alignment for Generation

> **Ngôn ngữ slide:** Tiếng Anh  
> **Ngôn ngữ thuyết trình:** Tiếng Việt  
> **Thời lượng ước tính:** 25–30 phút  
> **Số slide:** 28 slide chính + 3 slide ablation + 13 slide phụ lục (A–M)

---

## Slide 1: Title Slide

Xin chào mọi người. Hôm nay mình sẽ trình bày về paper **"Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think"**, viết tắt là **REPA**. Đây là paper được nhận Oral tại ICLR 2025, đến từ nhóm nghiên cứu tại KAIST, Korea University, Scaled Foundations và New York University.

---

## Slide 2: Outline

Bài trình bày sẽ gồm các phần: đầu tiên là giới thiệu và động lực nghiên cứu, tiếp theo là các quan sát chính, rồi đến phương pháp REPA, phân tích về cách REPA lấp đầy khoảng cách biểu diễn, kết quả thực nghiệm, và cuối cùng là kết luận cùng phần phụ lục.

---

## Slide 3: Motivation

Diffusion Transformers — cụ thể là DiT và SiT — đang là kiến trúc tiên tiến nhất cho bài toán sinh ảnh, đạt kết quả state-of-the-art trên ImageNet. Tuy nhiên, vấn đề lớn nhất là **chi phí huấn luyện cực kỳ đắt đỏ**. Ví dụ, mô hình SiT-XL/2 cần đến **7 triệu bước** để hội tụ.

Nhóm tác giả nhận ra rằng **nút thắt cổ chai** nằm ở việc mô hình phải tự học các biểu diễn nội tại chất lượng cao. Ý tưởng chính của paper là: thay vì để mô hình tự mày mò, ta sử dụng **các encoder thị giác đã được pretrain** — như DINOv2 — để hướng dẫn quá trình học biểu diễn. Kết quả? Chỉ cần một kỹ thuật đơn giản là đã đạt được tốc độ hội tụ **nhanh hơn 17.5 lần**.

Biểu đồ bên phải cho thấy REPA đạt cùng mức FID chỉ trong một phần nhỏ số bước huấn luyện so với vanilla.

---

## Slide 4: Background — Flow Matching & Diffusion Transformers

Trước khi đi vào phương pháp, mình sẽ review nhanh phần nền tảng.

**Flow Matching** là framework mà SiT sử dụng. Ý tưởng là: cho ảnh sạch $x_0$ và nhiễu $\epsilon$, ta tạo đầu vào nhiễu $x_t$ bằng phép nội suy: $x_t = \alpha_t \cdot x_0 + \sigma_t \cdot \epsilon$.

Với đường thẳng tuyến tính (linear path): $\alpha_t = 1 - t$ và $\sigma_t = t$, khi $t = 0$ ta có ảnh sạch, khi $t = 1$ ta có nhiễu hoàn toàn.

Mô hình được huấn luyện để dự đoán **velocity** $v_t = \dot{\alpha}_t \cdot x_0 + \dot{\sigma}_t \cdot \epsilon$. Đây gọi là **v-prediction**. Loss huấn luyện là MSE giữa velocity dự đoán và velocity thực.

Về kiến trúc, **DiT** thay thế U-Net bằng Transformer, và **SiT** mở rộng DiT với Scalable Interpolant framework. Cấu trúc gồm: patch embedding, các transformer block với adaLN-Zero conditioning, rồi unpatchify ra ảnh.

---

## Slide 5: Background — Self-Supervised Visual Encoders

Phần nền tảng thứ hai là về các **encoder tự giám sát** (self-supervised learning). Hiện nay có hai nhóm chính:

- **Contrastive / Self-distillation**: DINOv2, MoCo v3, CLIP — học biểu diễn bằng cách so sánh các view khác nhau của cùng ảnh.
- **Masked prediction**: MAE, I-JEPA — học bằng cách dự đoán phần bị che.

Câu hỏi đặt ra là: *Liệu ta có thể tận dụng các biểu diễn chất lượng cao này để giúp huấn luyện mô hình diffusion không?* Đây chính là ý tưởng cốt lõi của REPA — **align** biểu diễn của mô hình diffusion với biểu diễn từ encoder đã pretrain.

---

## Slide 6: Observation 1 — Diffusion Models Learn Representations

Bây giờ đến các quan sát quan trọng.

**Quan sát 1**: Mô hình diffusion *thực sự* học được biểu diễn có ý nghĩa phân biệt (discriminative). Khi thực hiện **linear probing** trên từng layer của SiT-XL/2 đã pretrain 7 triệu bước, ta thấy các layer giữa đến cuối cho accuracy khá cao.

Tuy nhiên, chất lượng biểu diễn này **thua xa** DINOv2. Biểu đồ bên phải cho thấy rõ khoảng cách này — đường DINOv2 nằm cao hơn hẳn SiT ở mọi layer. Vậy là có một **representation gap** đáng kể.

---

## Slide 7: Observation 2 — Weak Alignment

**Quan sát 2**: Nhóm tác giả đo **Centered Kernel Alignment (CKA)** giữa biểu diễn của SiT và DINOv2. CKA đo mức độ tương đồng giữa hai tập biểu diễn.

Kết quả cho thấy alignment rất **yếu** ở hầu hết các layer, đặc biệt ở các layer đầu thì gần như không có alignment. Các layer cuối có alignment khá hơn nhưng vẫn chưa tốt.

Điều này có nghĩa là: dù SiT có học được feature hữu ích, nhưng cách nó biểu diễn rất khác so với DINOv2.

---

## Slide 8: Observation 3 — Alignment Improves over Training

**Quan sát 3** — và đây là quan sát quan trọng nhất: Khi đo CKA qua các giai đoạn huấn luyện khác nhau, ta thấy alignment **liên tục cải thiện** khi train lâu hơn. Mô hình lớn hơn cũng align nhanh hơn.

Điều này gợi ý rằng: alignment với biểu diễn SSL là **xu hướng tự nhiên** của mô hình diffusion, chỉ là nó diễn ra **rất chậm**. Vậy nếu ta chủ động thúc đẩy quá trình này, huấn luyện sẽ hội tụ nhanh hơn rất nhiều!

---

## Slide 9: Summary of Observations

Tóm tắt 3 quan sát:
1. Mô hình diffusion *có* học biểu diễn, nhưng kém hơn SSL encoder
2. Alignment giữa diffusion model và DINOv2 rất yếu
3. Alignment này tự nhiên cải thiện theo thời gian huấn luyện

Từ đó, ý tưởng REPA ra đời: **chủ động align** biểu diễn ẩn của diffusion model với biểu diễn từ SSL encoder.

---

## Slide 10: REPA Overview

Đây là sơ đồ tổng quan của **REPA**. Ý tưởng rất đơn giản:

- Cho ảnh sạch $x_0$, ta đưa qua **encoder pretrain** (ví dụ DINOv2, frozen) để lấy biểu diễn sạch $z$.
- Đồng thời, ảnh nhiễu $x_t$ đi qua diffusion transformer. Tại một layer trung gian (layer $l$), ta trích xuất hidden state $h^{(l)}$.
- Dùng một **MLP projector** để chiếu $h^{(l)}$ vào cùng không gian với $z$, rồi **tối ưu cosine similarity** giữa hai biểu diễn.

Đó là toàn bộ phương pháp — một **regularization đơn giản** nhưng hiệu quả cực kỳ mạnh mẽ.

---

## Slide 11: Flow Matching Formulation

Phần này trình bày chi tiết hơn về công thức.

**Stochastic interpolant**: $x_t = \alpha_t \cdot x_0 + \sigma_t \cdot \epsilon$

Hai loại đường (path):
- **Linear**: $\alpha_t = 1-t$, $\sigma_t = t$ — đơn giản nhất
- **Cosine**: $\alpha_t = \cos(\pi t / 2)$, $\sigma_t = \sin(\pi t / 2)$ — mượt hơn

**Velocity target** cho v-prediction: $v_t = \dot{\alpha}_t \cdot x_0 + \dot{\sigma}_t \cdot \epsilon$

**Denoising loss**: $\mathcal{L}_{\text{denoise}} = \mathbb{E}[\|v_\theta(x_t, t) - v_t\|^2]$ — đây là loss tiêu chuẩn của flow matching, huấn luyện mô hình dự đoán velocity.

---

## Slide 12: REPA Alignment Loss

Đây là **loss chính** mà REPA thêm vào. Cụ thể:

1. Lấy feature sạch từ encoder: $z = f_{\text{enc}}(x_0)$, đây là các patch token.
2. Tại layer $l$ của diffusion model, lấy hidden state $h^{(l)}$ từ ảnh nhiễu.
3. Chiếu qua projector MLP: $\tilde{z} = f_{\text{proj}}(h^{(l)})$

**Alignment loss** là **negative cosine similarity** trung bình trên tất cả patch token:

$$\mathcal{L}_{\text{align}} = -\frac{1}{N}\sum_j \langle \bar{\tilde{z}}_j, \bar{z}_j \rangle$$

Trong đó $\bar{z}$ là vector đã normalize. Ý tưởng là: ép biểu diễn ẩn của diffusion model — dù nhận đầu vào nhiễu — phải "nhìn thấy" cùng thông tin ngữ nghĩa mà DINOv2 nhìn thấy từ ảnh sạch.

---

## Slide 13: Total Objective

**Loss tổng hợp** rất đơn giản:

$$\mathcal{L} = \mathcal{L}_{\text{denoise}} + \lambda \cdot \mathcal{L}_{\text{align}}$$

Với $\lambda = 0.5$ là giá trị mặc định.

**Projector MLP** gồm 3 layer Linear với 2 activation SiLU xen kẽ. Kích thước: từ $d$ (hidden size của transformer, ví dụ 1152 cho XL) → 2048 → 2048 → $d_z$ (kích thước embedding của encoder, ví dụ 768 cho DINOv2-ViT-B).

Điểm quan trọng: projector chỉ dùng khi **huấn luyện**, khi inference thì bỏ đi hoàn toàn. Tức là kiến trúc mô hình lúc sinh ảnh không thay đổi gì so với vanilla.

---

## Slide 14: Design Choices

Một số lựa chọn thiết kế quan trọng:

**Layer nào để align?** Mặc định là layer 8, tức khoảng 1/3 từ đầu cho SiT-XL/2 (28 block tổng cộng). Chọn layer sớm vì muốn các layer đầu nhanh chóng có biểu diễn semantic tốt.

**Encoder nào?** DINOv2 cho kết quả tốt nhất — cả ViT-B, ViT-L và ViT-g đều hoạt động. Nhưng các encoder khác như CLIP, MoCo v3, I-JEPA, MAE, DINOv1 cũng đều cải thiện so với vanilla.

**Nguyên lý thiết kế chính**: Align **layer đầu** với encoder SSL → giải phóng **layer sau** để tập trung vào chi tiết tần số cao phục vụ generation. Đây là sự phân công lao động tự nhiên giữa "hiểu" và "vẽ".

Và lưu ý: encoder hoàn toàn **frozen**, không backprop qua encoder; $\lambda = 0.5$; optimizer AdamW với learning rate $10^{-4}$.

---

## Slide 15: How REPA Works — Layer Specialization

Slide này minh họa cơ chế hoạt động.

**Không có REPA (Vanilla)**: Tất cả các layer phải đồng thời học cả biểu diễn ngữ nghĩa lẫn chi tiết sinh ảnh → hội tụ chậm vì các layer đầu mất rất lâu để học representation tốt.

**Có REPA**: Các layer đầu (block 1–8) được align nhanh chóng với DINOv2 → semantic representation được "tiêm" vào. Các layer sau (block 9–28) được giải phóng để tập trung vào xử lý tần số cao và chất lượng sinh ảnh.

Hình bên phải minh họa: DINOv2 frozen align trực tiếp vào khối layer đầu, tạo ra sự chuyên biệt hóa tự nhiên.

---

## Slide 16: Linear Evaluation — REPA vs Vanilla

Bây giờ chúng ta xem REPA lấp khoảng cách biểu diễn như thế nào.

Biểu đồ so sánh **linear probing accuracy** theo từng layer giữa vanilla SiT và SiT + REPA. REPA cải thiện **đáng kể** chất lượng biểu diễn ở các **layer đầu**, đưa chúng lên gần ngang tầm với DINOv2. Điều này xác nhận rằng alignment đã được chuyển giao thành công.

---

## Slide 17: CKA Alignment — REPA vs Vanilla

Tiếp theo, CKA alignment với DINOv2. REPA đạt alignment **mạnh hơn rất nhiều**, đặc biệt tại layer 8 (target layer). Các layer sau vẫn khác — và điều này là tốt, vì chúng cần chuyên biệt cho generation.

---

## Slide 18: Representation Slope

**Representation slope** đo mức độ mỗi layer thay đổi biểu diễn. Với REPA, ta thấy có một **chuyển tiếp rõ ràng** tại layer $l$: các layer trước layer 8 thay đổi mượt mà (aligned representations), trong khi các layer sau layer 8 thay đổi mạnh hơn (đang xử lý chi tiết cho generation).

Kết luận: REPA tạo ra sự **phân chia lao động** tự nhiên giữa phần mã hóa ngữ nghĩa và phần giải mã sinh ảnh.

---

## Slide 19: 17.5× Faster Training

Bây giờ đến phần kết quả thực nghiệm — và đây là phần ấn tượng nhất.

Biểu đồ cho thấy REPA match được performance của vanilla SiT-XL/2 train 7 triệu bước — chỉ trong **dưới 400K bước**. Đó là speedup **17.5 lần**! Cải thiện ổn định trên cả FID, Inception Score, và linear probing metrics.

---

## Slide 20: Qualitative Progression

Đây là so sánh trực quan trong quá trình huấn luyện. Hàng trên là **vanilla**, hàng dưới là **REPA**, cùng noise, cùng sampler, cùng số bước sampling.

Có thể thấy rõ: REPA tạo ra ảnh nhận diện được sớm hơn rất nhiều trong quá trình huấn luyện. Ở 50K bước, vanilla vẫn là nhiễu nhưng REPA đã bắt đầu có hình dạng rõ ràng.

---

## Slide 21: Results without CFG

Bảng kết quả **không dùng Classifier-Free Guidance**. REPA đạt **FID = 7.9** chỉ sau 400K bước, vượt mặt vanilla SiT-XL/2 train 7M bước (FID = 9.9). Cải thiện cũng thấy ở Inception Score và cả linear probing accuracy.

---

## Slide 22: Results with CFG

Khi kết hợp với **Classifier-Free Guidance** và **guidance interval scheduling**, REPA đạt **FID = 1.42** — đây là **state-of-the-art** trên ImageNet 256×256 tại thời điểm công bố. Đặc biệt, kết quả này đạt được với **ít hơn 7 lần** số epoch so với các phương pháp trước đó.

---

## Slide 23: Encoder Scalability

Bảng này cho thấy kết quả khi thay đổi **encoder pretrain**. Quy luật rõ ràng: encoder tốt hơn → generation tốt hơn VÀ representation tốt hơn. Gia đình DINOv2 cho kết quả tốt nhất, nhưng ngay cả các encoder yếu hơn như MAE hay DINOv1 cũng cải thiện đáng kể so với vanilla.

---

## Slide 24: Scaling Laws

Hai biểu đồ về scaling:

**Bên trái**: REPA mang lại **speedup lớn hơn cho mô hình lớn hơn** — tức là REPA scale rất tốt.

**Bên phải**: Có **tương quan log-linear** giữa linear probing accuracy và FID. Điều này rất quan trọng — nó chứng minh rằng cải thiện representations TRỰC TIẾP dẫn đến cải thiện generation quality. Đây là insight cốt lõi của paper.

---

## Slide 25: Qualitative Samples

Đây là các ảnh mẫu sinh ra từ SiT-XL/2 + REPA. Có thể thấy chất lượng rất cao, chi tiết sắc nét, đa dạng class.

---

## Slide 26: Key Takeaways

Tóm tắt các điểm chính:

1. **Chất lượng biểu diễn là nút thắt cổ chai** trong huấn luyện diffusion transformers.
2. **REPA** là regularization đơn giản: align hidden states với SSL encoder bằng cosine similarity.
3. Kết quả rất ấn tượng: 17.5 lần nhanh hơn, FID = 1.42 SOTA, hoạt động với nhiều encoder.
4. REPA tạo ra **sự chuyên biệt hóa layer**: layer đầu cho ngữ nghĩa, layer sau cho chi tiết.
5. Takeaway thực tiễn: huấn luyện diffusion transformer mạnh mẽ giờ đây **dễ dàng và rẻ hơn nhiều**.

---

## Slide 27: Limitations & Future Work

**Hạn chế:**
- Cần encoder pretrain — thêm dependency
- Chủ yếu demo trên ImageNet — cần nghiên cứu thêm trên domain đa dạng hơn
- Projector MLP thêm overhead nhỏ khi train (nhưng bỏ đi khi inference)

**Hướng phát triển:**
- Áp dụng REPA cho **text-to-image** ở quy mô lớn (đã có kết quả ban đầu trên MS-COCO)
- Kết hợp với **multi-modal encoder** để hỗ trợ text conditioning
- Kết hợp với progressive training hoặc distillation
- Nghiên cứu adaptive alignment depth — tự động chọn layer tối ưu

---

## Slide 28: Thank You

Cảm ơn mọi người đã lắng nghe. Mọi thông tin chi tiết có thể tham khảo tại paper trên arXiv, source code trên GitHub, và project page. Mình sẵn sàng nhận câu hỏi.

---

## Slide A1: Training Hyperparameters (Phụ lục)

Đây là bảng tổng hợp các hyperparameter huấn luyện. Đáng chú ý: optimizer AdamW với learning rate cố định $10^{-4}$, batch size 256, mixed precision FP16. Các tham số đặc trưng của REPA là $\lambda = 0.5$ cho alignment coefficient, encoder depth là 8, và projector hidden dimension là 2048.

---

## Slide A2: Model Configurations (Phụ lục)

Bảng này liệt kê cấu hình của các model SiT (B, L, XL) cùng các encoder DINOv2. SiT-XL/2 — mô hình chính trong paper — có 28 layer, hidden size 1152, khoảng 675 triệu tham số. Ảnh 256×256 được encode thành latent 32×32 qua VAE, chia thành 256 patch token với patch size 2.

---

## Slide A3: Projector MLP Architecture (Phụ lục)

Sơ đồ chi tiết của Projector MLP: 3 layer Linear xen kẽ 2 activation SiLU. Kích thước cụ thể cho SiT-XL với DINOv2-ViT-B: 1152 → 2048 → 2048 → 768. Projector này chỉ tồn tại trong quá trình huấn luyện; khi inference, nó được loại bỏ hoàn toàn nên không ảnh hưởng đến tốc độ sinh ảnh.

---

## Slide A4: Ablation — Which Layer to Align? (Phụ lục)

Bảng này cho thấy kết quả ablation khi thay đổi **encoder depth** $l$ — tức là layer mà REPA thực hiện alignment. Kết quả cho thấy layer 8 là điểm tối ưu cho SiT-XL/2 (28 block). Nếu align quá nông ($l=2$), mô hình chưa có đủ capacity trước điểm alignment. Nếu quá sâu ($l=16$), các layer phía sau không còn đủ không gian để xử lý chuyên biệt cho sinh ảnh. Quy tắc chung là align ở khoảng 25–30% tổng depth.

---

## Slide A5: Ablation — Alignment Coefficient $\lambda$ (Phụ lục)

Ablation tiếp theo là về hệ số alignment $\lambda$. Khi $\lambda = 0$ chính là vanilla (không có REPA). Tăng $\lambda$ lên 0.5 cho kết quả tốt nhất. Đáng chú ý, ngay cả $\lambda$ nhỏ (0.1) cũng đã cải thiện đáng kể so với vanilla. Tuy nhiên lambda quá lớn (2.0) lại gây hại vì alignment chiếm ưu thế quá mức so với denoising loss. REPA khá robust trong khoảng $\lambda \in [0.25, 1.0]$.

---

## Slide A6: Ablation — Loss Function Variants (Phụ lục)

Tại sao lại dùng cosine similarity thay vì MSE hay Smooth L1? Kết quả cho thấy cosine similarity cho FID tốt nhất. Lý do là: normalization làm loss không phụ thuộc vào scale, tập trung vào hướng alignment thay vì biên độ. Điều này giúp huấn luyện ổn định hơn, tránh mismatch giữa các không gian encoder khác nhau. Cách tiếp cận này cũng nhất quán với objective của các phương pháp self-supervised learning như DINO và BYOL.

---

## Slide A7: Text-to-Image Generation — Appendix K (Phụ lục)

Đây là kết quả từ **Appendix K** của paper. Nhóm tác giả thử nghiệm REPA trên bài toán **text-to-image** với dataset MS-COCO. Kiến trúc sử dụng là **MMDiT** — một biến thể của DiT xử lý đồng thời image patches và text embeddings. Mô hình có hidden dim 768, depth 24, CLIP text encoder, và được train 150K iterations.

Kết quả rất rõ ràng: với ODE sampler, FID giảm từ 6.05 xuống còn **4.73**; với SDE sampler, giảm từ 5.30 xuống còn **4.14** — tốt hơn cả U-ViT-S/2 (Deep) được train cùng điều kiện. Điều này xác nhận REPA là phương pháp **kiến trúc-agnostic và task-agnostic**: bất kỳ denoising model nào cũng hưởng lợi từ alignment biểu diễn, dù có text conditioning hay không.

---

## Slide A8: ImageNet 512×512 — Appendix J (Phụ lục)

Đây là kết quả từ **Appendix J**. Setup hoàn toàn giống 256×256, chỉ khác input dimension: SD-VAE encode ảnh 512×512 thành latent 64×64×4, tức 1024 patch token. DINOv2 nhận ảnh resize lên 448×448 với positional embedding interpolation.

Table 11 cho thấy kết quả rất ấn tượng: SiT-XL/2 + REPA chỉ cần **200 epochs** để vượt qua vanilla SiT-XL/2 train **600 epochs** — FID giảm từ 2.62 xuống còn **2.08**, IS tăng từ 252 lên **274.6**. Tức là REPA tiết kiệm hơn **3 lần** số epoch mà vẫn cho kết quả tốt hơn. Ngay cả ở 80 epochs, REPA đã cạnh tranh được với baseline 600 epochs của baseline.

---

## Slide A9: Sampling Methods & Guidance (Phụ lục)

Cuối cùng, REPA hoạt động với nhiều phương pháp sampling khác nhau. Kết quả FID tốt nhất (1.42) đạt được với **SDE sampler**, CFG scale 1.8 và **guidance interval** $[0, 0.7]$. Guidance interval là kỹ thuật chỉ áp dụng CFG trong một khoảng noise nhất định, giúp giảm artifact từ over-guidance ở mức noise thấp. Euler ODE và Heun ODE cũng cho kết quả cạnh tranh.

---

*Hết script.*
