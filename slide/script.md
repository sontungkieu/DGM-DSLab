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

Bài trình bày sẽ gồm các phần: đầu tiên là giới thiệu và động lực nghiên cứu, tiếp theo là các quan sát chính, rồi đến phương pháp REPA, phân tích về cách REPA cải thiện biểu diễn, kết quả thực nghiệm, và cuối cùng là kết luận cùng phần phụ lục.

---

## Slide 3: Motivation

Diffusion Transformers — cụ thể là DiT và SiT — đang là những kiến trúc tiên tiến nhất cho bài toán sinh ảnh, đạt kết quả state-of-the-art trên ImageNet. Tuy nhiên, vấn đề lớn nhất là **chi phí huấn luyện cực kỳ đắt đỏ**. Ví dụ, mô hình SiT-XL/2 cần đến **7 triệu bước** để hội tụ.

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

**Quan sát 2**: Nhóm tác giả đo **CKNNA** (Centered Kernel Nearest-Neighbor Alignment, Huh et al. 2024) — một biến thể của CKA dùng k-nearest neighbors — giữa biểu diễn của SiT và DINOv2. CKNNA đo mức độ tương đồng cục bộ giữa hai tập biểu diễn.

Kết quả cho thấy alignment rất **yếu** ở hầu hết các layer, đặc biệt ở các layer đầu thì gần như không có alignment. Các layer cuối có alignment khá hơn nhưng vẫn chưa tốt.

Điều này có nghĩa là: dù SiT có học được feature hữu ích, nhưng cách nó biểu diễn rất khác so với DINOv2.

---

## Slide 8: Observation 3 — Alignment Improves over Training

**Quan sát 3** — và đây là quan sát quan trọng nhất: Khi đo **CKNNA** qua các giai đoạn huấn luyện khác nhau, ta thấy alignment **liên tục cải thiện** khi train lâu hơn. Mô hình lớn hơn cũng align nhanh hơn.

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

## Slide 17: CKNNA Alignment — REPA vs Vanilla

Tiếp theo, **CKNNA** alignment với DINOv2. REPA đạt alignment **mạnh hơn rất nhiều**, đặc biệt tại layer 8 (target layer). Các layer sau vẫn khác — và điều này là tốt, vì chúng cần chuyên biệt cho generation.

---

## Slide 18: Bridging the Gap — Generation–Representation Frontier (Figure 3c)

Slide này trình bày **Figure 3c** của paper — biểu đồ minh họa trực tiếp mối liên hệ giữa chất lượng biểu diễn và chất lượng sinh ảnh.

Trục X là **Validation Accuracy** (linear probing, tăng dần là tốt hơn), trục Y là **FID-50K** (giảm dần là tốt hơn). Mỗi điểm trên biểu đồ ứng với một checkpoint tại các mốc training (50K, 100K, 200K, 400K iterations).

Với **vanilla SiT-XL/2**: các điểm di chuyển rất chậm trên cả hai trục — mô hình cải thiện rất ít cả về representation lẫn generation sau cùng số bước.

Với **SiT-XL/2 + REPA**: các điểm nhảy vọt nhanh chóng lên góc trên trái — **FID thấp hơn và Accuracy cao hơn** cùng lúc trong cùng số iterations.

Kết luận của nhóm tác giả: REPA **đẩy đường biên generation–representation** (the envelope) — tức là cải thiện alignment biểu diễn trực tiếp dẫn đến cải thiện chất lượng sinh ảnh, theo cách đồng thời và nhất quán.

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

Bảng kết quả **không dùng Classifier-Free Guidance**. REPA đạt **FID = 7.9** chỉ sau 400K bước, vượt mặt vanilla SiT-XL/2 train 7M bước (FID = 8.3). Cải thiện cũng thấy ở Inception Score và cả linear probing accuracy.

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

**Bên phải**: Tương quan âm rõ ràng giữa linear probing accuracy và FID — accuracy cao hơn tức là FID thấp hơn, tức là biểu diễn tốt hơn trực tiếp dẫn đến generation tốt hơn. Các mô hình lớn hơn thể hiện **cải thiện dốc hơn** trên cả hai chỉ số khi train lâu hơn — đây là insight cốt lõi của paper: đây không phải tương quan ngẫu nhiên mà là quan hệ nhân quả.

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

## Slide Abl-1: Ablation — Alignment Depth (Phụ lục Ablation)

Slide này trình bày kết quả ablation đầu tiên: **tác động của việc chọn layer để align**, tức là giá trị $l$ (encoder depth). Thực nghiệm được thực hiện trên SiT-L/2, train 400K vòng lặp với DINOv2-L encoder và NT-Xent loss.

Kết quả bảng 2 cho thấy **layer 8 là điểm tối ưu**, đạt FID = 10.0. SiT-L/2 có tổng cộng 24 block, tức layer 8 tương đương khoảng một phần ba đầu của mô hình. Nếu align quá nông — ví dụ layer 6 — FID tăng lên 10.3, vì các layer quá đầu chưa có đủ capacity để tiếp nhận tín hiệu semantic. Ngược lại, nếu align quá sâu — layer 16 — FID lên tới 12.1, vì ta đang ép quá nhiều layer phía sau phải làm việc với semantic thay vì chuyên biệt hóa cho sinh ảnh chi tiết.

Một quan sát thú vị: **linear probing accuracy lại tăng khi align sâu hơn** (68.1 → 71.1), nhưng FID lại xấu hơn. Điều này xác nhận giả thuyết của paper: layers sau cần được "tự do" để học đặc trưng tần số cao phục vụ sinh ảnh — không phải để học semantics.

---

## Slide Abl-2: Ablation — Lambda (Phụ lục Ablation)

Ablation thứ hai là về **hệ số alignment** $\lambda$ — kiểm soát mức độ đóng góp của alignment loss so với denoising loss. Thực nghiệm với SiT-XL/2, 400K iterations.

Bảng 5 cho thấy: $\lambda = 0.25$ cho FID = 8.6, tăng lên $\lambda = 0.5$ thì FID giảm xuống **7.9** — đây là mức mặc định được chọn. Tuy nhiên, kết quả **gần như bão hòa** từ $\lambda = 0.5$ trở lên: $\lambda = 0.75$ cho FID 7.8, $\lambda = 1.0$ cũng cho FID 7.8. IS tăng liên tục nhưng không đáng kể.

Điều này cho thấy REPA rất **robust với $\lambda$**: miễn là có alignment, mô hình cải thiện rõ ràng so với vanilla (FID was 17.2). Nhóm tác giả chọn $\lambda = 0.5$ vì nó nằm đúng ngưỡng bão hòa đồng thời còn dư địa cho denoising loss chiếm ưu thế.

---

## Slide Abl-3: Ablation — Alignment Objective (Phụ lục Ablation)

Ablation cuối cùng là về **loại objective được dùng để alignment**: NT-Xent (contrastive) hay cosine similarity (từng patch).

Kết quả từ bảng 2 (SiT-L/2, depth=8, 400K iterations): NT-Xent đạt FID = 10.0, cosine similarity đạt FID = **9.9**, gần tương đương. Tuy nhiên cosine similarity có IS cao hơn (111.9 vs 106.6), đồng thời đơn giản hơn về mặt implementation.

Nhóm tác giả lưu ý rằng NT-Xent có lợi thế ở **giai đoạn đầu training** (50–100K iterations) vì nó cung cấp gradient mạnh hơn qua contrastive signal. Nhưng sau 400K iterations, cosine similarity kịp bắt kịp và thậm chí nhỉnh hơn. Do vậy, **cosine similarity được chọn làm mặc định** vì tính đơn giản và hiệu quả tương đương.

---

## Slide App-A: Appendix A — Diffusion-Based Models (Phụ lục A)

**Appendix A** mô tả hai framework sinh ảnh được sử dụng trong paper.

Framework đầu tiên là **DiT** (Peebles & Xie, 2023) dựa trên DDPM — mô hình phổ biến nhất trong dòng diffusion. DDPM học nghịch quá trình thêm nhiễu Gaussian dần dần, với objective là dự đoán nhiễu $\epsilon$. DiT thay thế U-Net bằng Transformer với patch size 2.

Framework thứ hai là **SiT** — nền tảng chính của paper — sử dụng **stochastic interpolants**. Điểm quan trọng: SiT học dự đoán **velocity** $v$ thay vì noise $\epsilon$, và dùng sampler Euler-Maruyama với 250 function evaluations. Một chi tiết kỹ thuật quan trọng là last step của SDE sampler được đặt là **0.04** — nhóm tác giả cho thấy điều này cho cải thiện đáng kể chất lượng sinh ảnh.

REPA áp dụng được cho cả hai framework, nhưng kết quả chính đều dùng SiT vì SDE sampler cho chất lượng cao hơn.

---

## Slide App-B: Appendix B — Architecture (Phụ lục B)

**Appendix B** mô tả chi tiết kiến trúc Diffusion Transformer. Bảng Table 1 liệt kê ba cấu hình:
- B/2: 12 layer, hidden 768, 12 attention heads
- L/2: 24 layer, hidden 1024, 16 heads
- **XL/2**: 28 layer, hidden 1152, 16 heads — mô hình chính trong paper với khoảng 675 triệu tham số

Tất cả đều dùng patch size 2, ảnh 256×256 qua VAE thành latent 32×32×4, cho ra **256 patch tokens**.

Mỗi DiT block gồm: adaLN-Zero để modulate hidden state theo timestep và class embedding, self-attention, feed-forward MLP, với skip connection và layer norm. **REPA projection** là một MLP 3 layer gắn sau layer $l$ đầu tiên của mô hình — chỉ tồn tại trong lúc train, bị bỏ đi khi inference.

---

## Slide App-C: Appendix C — CKNNA Metric (Phụ lục C)

**Appendix C** định nghĩa chính thức metric **CKNNA** — Centered Kernel Nearest-Neighbor Alignment — được dùng xuyên suốt paper để đo alignment.

CKNNA là phiên bản mở rộng của CKA cổ điển (Kornblith et al., 2019). CKA truyền thống dùng HSIC để đo độ tương đồng toàn cục giữa hai tập representation. CKNNA thay HSIC bằng Align(K,L) — chỉ tính trên các cặp **k láng giềng gần nhất** (k-nearest neighbors) trong mỗi không gian feature. Điều này làm cho CKNNA **nhạy cảm hơn** với cấu trúc cục bộ của không gian biểu diễn.

Trong paper: $k = 10$, dùng **10,000 ảnh** từ ImageNet val. Linear probing train 90 epochs với Adam optimizer, cosine decay, learning rate $10^{-3}$.

---

## Slide App-D: Appendix D — Hyperparameters (Phụ lục D)

**Appendix D** tổng hợp toàn bộ hyperparameter.

**Phần huấn luyện chung**: AdamW optimizer với learning rate không đổi $10^{-4}$, $(\beta_1, \beta_2) = (0.9, 0.999)$, weight decay = 0, batch size 256, mixed precision FP16 với gradient clipping. Khi evaluation: SDE Euler-Maruyama sampler với NFE=250, last SDE step = 0.04.

**Phần REPA-specific**: $\lambda = 0.5$, encoder depth $l = 8$, encoder mặc định là DINOv2-B, similarity measure là cosine similarity. Tổng số iterations: 400K (ngắn) hoặc 4M (đầy đủ).

**Tài nguyên tính toán**: 8 GPU NVIDIA H100 80GB, tốc độ khoảng 5.4 bước/giây với batch 256. VAE latent được tính trước để tiết kiệm thời gian.

---

## Slide App-E+F: Appendix E+F — Evaluation & Baselines (Phụ lục E+F)

**Appendix E** mô tả các metric được dùng: FID (đo khoảng cách phân phối qua Inception-v3), sFID (FID theo spatial features), IS (Inception Score), Precision & Recall (Kynkäänniemi et al., 2019), linear probing accuracy (Acc.), và CKNNA (alignment score).

**Appendix F** mô tả các baseline, chia làm 4 nhóm:
- **Pixel diffusion**: ADM, VDM++, Simple diffusion, CDM
- **Latent U-Net**: LDM
- **Transformer + U-Net hybrid**: U-ViT-H/2, DiffiT, MDTv2-XL/2
- **Pure latent transformer**: MaskDiT, SD-DiT, DiT-XL/2, SiT-XL/2

Tất cả đều dùng cùng VAE và protocol đánh giá để so sánh công bằng.

---

## Slide App-G: Appendix G — Detailed Results Without CFG (Phụ lục G)

**Appendix G** (Table 8) trình bày kết quả chi tiết theo từng scale model, **không dùng CFG**. Đây là bảng dữ liệu thực từ paper.

Vài điểm đáng chú ý:
- SiT-B/2 vanilla (400K): FID = 33.0; với REPA: FID = **24.4** — cải thiện đáng kể ngay cả model nhỏ
- SiT-L/2 vanilla (400K): FID = 18.8; với REPA: FID = **10.0** — giảm gần một nửa
- **SiT-XL/2 vanilla cần 7M iterations** để đạt FID = 8.3; **REPA chỉ cần 400K iterations** để đạt FID = **7.9** — tức là đã vượt qua với ít hơn 18 lần số bước
- Với 4M iterations, REPA SiT-XL/2 đạt FID = **5.9** — cải thiện mạnh thêm

Cột Acc. cũng cho thấy linear probing accuracy tăng liên tục với REPA ở mọi scale — từ 61.2% (B/2) đến 74.6% (XL/2 4M).

---

## Slide App-G2: Appendix G — Detailed Results With CFG (Phụ lục G)

Slide tiếp theo trong Appendix G là kết quả **với CFG và guidance interval** (Tables 9–10).

Nhóm tác giả thử nghiệm nhiều cấu hình guidance interval $[0, t_{high}]$ với $t_{high}$ khác nhau. Phát hiện: interval $[0, 0.7]$ với $w = 1.80$ cho kết quả tốt nhất — **FID = 1.42**, sFID = 4.70, IS = 305.7. Đây là kết quả state-of-the-art trên ImageNet 256×256 tại thời điểm công bố.

Lý do guidance interval hoạt động: áp dụng CFG ở noise level thấp (gần ảnh sạch) gây ra artifact. Khi chỉ áp dụng CFG cho $t \in [0, 0.7]$ — tức là giai đoạn coarse structure — mô hình vừa được hưởng lợi từ class conditioning vừa tránh được over-guidance ở giai đoạn fine detail.

---

## Slide App-J: Appendix J — ImageNet 512×512 (Phụ lục J)

**Appendix J** mở rộng REPA sang bài toán sinh ảnh **độ phân giải 512×512**.

Setup thay đổi so với 256×256: SD-VAE encode ảnh 512×512 thành latent 64×64×4, tức **1024 patch token**. DINOv2 nhận ảnh resize lên 448×448.

Table 11 (với CFG $w = 1.35$) cho thấy kết quả ấn tượng: **SiT-XL/2 + REPA chỉ cần 200 epochs** để đạt FID = **2.08** — vượt qua vanilla SiT-XL/2 train đến 600 epochs (FID = 2.62). Tức là tiết kiệm hơn **3 lần số epoch** mà cho chất lượng tốt hơn. Ngay cả ở chỉ 80 epochs, REPA đã đạt FID = 2.44, cạnh tranh được với baseline 600 epochs.

Điều này xác nhận REPA scale tốt ra ngoài phạm vi 256×256.

---

## Slide App-K: Appendix K — Text-to-Image (Phụ lục K)

**Appendix K** thử nghiệm REPA trên bài toán **text-to-image generation** với dataset MS-COCO, sử dụng kiến trúc **MMDiT** (Multi-Modal DiT) — xử lý đồng thời image patches và text embeddings từ CLIP.

Setup: hidden dim 768, depth 24, train 150K iterations, batch 256, CFG $w = 2.0$.

Kết quả Table 12:
- ODE sampler: FID giảm từ **6.05 → 4.73** (cải thiện 22%)
- SDE sampler: FID giảm từ **5.30 → 4.14** (cải thiện 22%)

MMDiT + REPA với SDE đạt FID 4.14, tốt hơn cả U-ViT-S/2 Deep (FID 5.48). Kết quả này xác nhận REPA là phương pháp **kiến trúc-agnostic và task-agnostic** — nó cải thiện đáng kể ngay cả khi có thêm text conditioning.

---

## Slide App-L: Appendix L — Feature Map Visualization (Phụ lục L)

**Appendix L** (Figure 38) là phần visualization trực quan nhất của paper. Nhóm tác giả áp dụng **PCA** — tương tự như cách DINOv2 visualize features — để so sánh các feature map của SiT-XL/2 có và không có REPA.

Với **SiT-XL/2 + REPA**: ở các **layer đầu (1–6)**, feature map thể hiện ngay cấu trúc semantic cấp cao — ta có thể nhận ra hình dạng đối tượng. Sang các **layer sau (14–28)**, features chuyển sang chi tiết cục bộ, texture và tần số cao. Đây là kiểu pattern **coarse-to-fine** rất rõ ràng ở tất cả các timestep $t$.

Với **vanilla SiT-XL/2**: features trở nên **nhiễu** ở timestep lớn ($t = 0.7, 0.9$), không có cấu trúc phân cấp rõ ràng.

Bức ảnh minh họa này (Figure 38 từ paper) trực tiếp chứng minh **REPA tạo ra layer specialization**: early layers học semantics, later layers tự do chuyên biệt hóa cho generation.

---

## Slide App-M: Appendix M — Limitations & Future Work (Phụ lục M)

Slide cuối cùng của phụ lục — **Appendix M** — nêu các hạn chế hiện tại và bốn hướng nghiên cứu tương lai.

**Ba hạn chế chính**: REPA đòi hỏi phải có pretrained SSL encoder sẵn sàng, hiện chủ yếu được kiểm chứng trên ImageNet với class-conditional generation, và projector MLP thêm một lượng nhỏ overhead trong huấn luyện (dù bị bỏ đi lúc inference).

**Bốn hướng tương lai** (theo Appendix M):
1. **Phân tích lý thuyết** về tại sao layer 8 là tối ưu — hiện tại vẫn chỉ là empirical
2. **Mở rộng sang dữ liệu đa dạng hơn**: pixel-level, video, text-to-image ở quy mô lớn
3. **Phân tích lý thuyết** về mối liên hệ giữa instance discrimination và denoising
4. **Time-varying REPA** — hàm $\lambda(t)$ thay đổi theo noise schedule, có thể cho alignment tốt hơn ở từng timestep

Tổng kết lại: REPA là một phương pháp đơn giản nhưng hiệu quả, mở ra nhiều hướng nghiên cứu mới về intersection giữa representation learning và generative modeling.

---

*Hết script.*
