# REPA Presentation Script (Vietnamese)

## Slide 1 - Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think
Mục tiêu: Mở bài, giới thiệu paper, nêu rõ câu hỏi trung tâm của buổi trình bày.

Lời thoại:  
Ở slide mở đầu này, mình muốn đặt khung cho cả bài nói. Paper mình trình bày có tên là "Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think", của Yu và các cộng sự, xuất hiện ở ICLR 2025. Câu hỏi chính của paper là: tại sao diffusion transformer lại cần rất nhiều bước train mới đạt chất lượng cao, và liệu điểm nghẽn thực sự có nằm ở khả năng học biểu diễn hay không. Tác giả cho rằng bài toán không chỉ là dự đoán noise hay velocity cho đúng, mà còn là học được một hidden representation đủ mạnh về mặt ngữ nghĩa. Từ góc nhìn đó, paper đề xuất một regularization rất đơn giản tên là REPA, dùng encoder pretrained để ép hidden state của model khuếch tán thẳng hàng với semantic feature sạch của ảnh gốc.

Chuyển slide:  
Với khung đó, slide tiếp theo sẽ đi thẳng vào luận điểm cốt lõi của paper chỉ trong một trang.

Nhấn mạnh / phát âm:  
REPA đọc là "ree-pa". DINOv2 đọc là "dai-no version two".

## Slide 2 - Thesis: make representation learning explicit
Mục tiêu: Tóm tắt luận đề trung tâm và headline result.

Lời thoại:  
Nếu phải nén toàn bộ paper vào một ý, thì ý đó là: đừng để diffusion transformer phải tự mày mò học ngữ nghĩa chỉ từ denoising loss. REPA biến phần học biểu diễn thành một mục tiêu rõ ràng bằng cách lấy hidden state sớm của model và align nó với feature sạch từ một encoder pretrained như DINOv2. Khi làm như vậy, hidden state trở nên có nghĩa sớm hơn trong quá trình học. Điều đó giải phóng các layer phía sau để tập trung vào phần thật sự mang tính generative hơn, ví dụ texture, cấu trúc cục bộ, hay chi tiết tần số cao. Hệ quả rất quan trọng là mô hình không chỉ tốt hơn ở cuối đường train, mà còn đi tới vùng chất lượng tốt nhanh hơn rất nhiều.

Chuyển slide:  
Để thấy vì sao ý tưởng này hợp lý, mình cần nói rõ điểm nghẽn mà paper nhắm tới là gì.

## Slide 3 - Why representation quality is the bottleneck
Mục tiêu: Giải thích trực giác vì sao denoising loss chưa đủ để học semantic feature tốt.

Lời thoại:  
Ở đây tác giả muốn đổi cách nhìn về diffusion training. Bình thường ta nghĩ model chỉ cần học nghịch đảo quá trình corruption là đủ. Nhưng thực tế hidden state bên trong model phải gánh hai việc cùng lúc: một là nắm được semantics của ảnh, tức là ảnh này là chó, chim, xe lửa hay bối cảnh trong nhà; hai là phục hồi chi tiết hình học và texture từ dữ liệu nhiễu. Vấn đề là objective denoising chỉ giám sát đầu ra tái tạo, nên semantic structure chỉ được học như một hệ quả phụ. Điều đó làm quá trình tối ưu tốn dữ liệu, tốn compute, và chậm. Paper vì thế đặt giả thuyết rằng điểm nghẽn lớn trong diffusion transformer không chỉ là inverse corruption, mà là internal representation learning.

Chuyển slide:  
Vậy bằng chứng nào cho thấy hidden state của diffusion model thực sự có ngữ nghĩa nhưng vẫn chưa đủ tốt.

## Slide 4 - Diffusion features are meaningful, but still misaligned
Mục tiêu: Trình bày bằng chứng thực nghiệm trước khi đưa ra phương pháp REPA.

Lời thoại:  
Slide này là phần thuyết phục quan trọng. Tác giả dùng hai nhóm đo lường. Thứ nhất là linear probing để kiểm tra hidden state có tách được class semantics hay không. Thứ hai là CKNNA để đo mức độ aligned giữa hidden state của diffusion transformer và feature từ encoder mạnh như DINOv2. Kết quả cho thấy hidden state của SiT không hề vô nghĩa. Nó có semantic signal thật, đặc biệt ở các layer sâu hơn. Nhưng mức alignment với DINOv2 vẫn khá yếu so với alignment giữa các mô hình self-supervised mạnh với nhau. Ý nghĩa của kết quả này là: diffusion model đúng là đang học biểu diễn, nhưng học chậm, học đắt, và học chưa hiệu quả.

Chuyển slide:  
Trước khi đi vào REPA, mình nhắc lại rất nhanh ký hiệu và góc nhìn stochastic interpolants mà paper dùng.

## Slide 5 - Preliminaries: stochastic interpolants viewpoint
Mục tiêu: Thiết lập ký hiệu cho dữ liệu sạch, noise, và biến thời gian.

Lời thoại:  
Paper trình bày phần nền tảng theo ngôn ngữ stochastic interpolants. Ta có một mẫu sạch là x sao, một noise Gaussian epsilon, và trạng thái trung gian x t được xây bằng tổ hợp alpha t nhân với dữ liệu sạch cộng sigma t nhân với noise. Khi t bằng 0 thì ta ở phía dữ liệu, còn khi t bằng 1 thì ta ở phía noise. Với SiT, lựa chọn rất đơn giản là alpha t bằng một trừ t và sigma t bằng t. Điểm mình muốn nhấn mạnh là REPA sẽ lấy hidden state sinh ra từ input đã bị nhiễu, nhưng target feature lại đến từ ảnh sạch. Nói cách khác, alignment ở đây là ép model phải giữ được semantic identity ngay cả khi đầu vào đã bị corruption.

Chuyển slide:  
Từ ký hiệu này, slide sau nhắc lại loss gốc mà REPA sẽ cộng thêm vào.

## Slide 6 - Training objective recap
Mục tiêu: Nhắc lại velocity loss và quan hệ giữa velocity với score.

Lời thoại:  
Ở phần này, paper nhắc lại objective gốc của SiT dưới dạng velocity prediction. Model học một vector field v của x và t sao cho khớp với đạo hàm của đường nội suy giữa dữ liệu sạch và noise. Từ velocity ta cũng suy ra score, và do đó suy ra reverse-time dynamics để sampling. Điều cần lưu ý là REPA không thay đổi phần toán học này. Tất cả công thức denoising vẫn giữ nguyên. REPA chỉ chèn thêm một lực kéo ở không gian biểu diễn: hidden state bên trong model không chỉ cần hữu ích cho việc dự đoán velocity, mà còn cần giống với semantic feature sạch do encoder pretrained cung cấp.

Chuyển slide:  
Bây giờ mình chuyển từ phần nền sang trực giác của phương pháp.

## Slide 7 - Intuition: noisy hidden states should predict clean semantics
Mục tiêu: Trình bày trực giác high-level của REPA.

Lời thoại:  
Trực giác của REPA rất gọn. Ta đưa ảnh sạch đi qua một encoder đóng băng, ví dụ DINOv2, để lấy target feature. Đồng thời, diffusion transformer nhận latent bị nhiễu và tạo ra hidden state ở một block trung gian. Sau đó, một projector nhỏ sẽ ánh xạ hidden state này sang cùng feature space với encoder pretrained. Loss alignment sẽ kéo hai biểu diễn này lại gần nhau theo từng patch. Điều hay ở đây là không cần đổi backbone, không cần đổi sampler, và cũng không cần pretrain diffusion model theo cách hoàn toàn mới. Paper tận dụng tri thức ngữ nghĩa vốn đã có sẵn trong encoder mạnh, rồi distill nó vào hidden state của generator.

Chuyển slide:  
Từ trực giác đó, slide kế tiếp viết objective ra công thức chính thức.

## Slide 8 - Formal objective
Mục tiêu: Trình bày loss REPA và loss tổng.

Lời thoại:  
Ở đây tác giả định nghĩa y sao là feature sạch từ encoder frozen, còn h t là hidden state của diffusion model. Projector h phi sẽ biến hidden state sang đúng chiều và đúng cấu trúc của target feature. Loss REPA về bản chất là tối đa hóa độ tương đồng giữa projected hidden state và clean feature, thường bằng cosine hoặc một biến thể contrastive như NT-Xent. Sau đó loss tổng chỉ đơn giản là loss denoising cộng lambda nhân với loss REPA. Điểm cần nhớ là target luôn đến từ ảnh sạch chứ không phải từ ảnh nhiễu. Chính quyết định này làm REPA trở thành một hình thức dạy model khôi phục semantics bất biến với corruption, chứ không chỉ là khớp thêm một representation bất kỳ.

Chuyển slide:  
Tiếp theo mình nói REPA được gắn vào kiến trúc DiT và SiT ở đâu.

## Slide 9 - Architecture: where REPA is attached
Mục tiêu: Mô tả vị trí gắn projector và hidden state alignment trong backbone.

Lời thoại:  
Slide này nhấn mạnh tính modular của phương pháp. Backbone vẫn là DiT hoặc SiT tiêu chuẩn: patchify latent, chạy qua chuỗi transformer blocks, rồi dự đoán mục tiêu denoising. REPA không đụng vào head chính. Nó chỉ "tap" một hidden state ở block trung gian, thường là block sớm, rồi đưa qua một MLP ba tầng để align với feature space của encoder ngoài. Vì projector chỉ dùng trong train, chi phí thêm vào inference gần như bằng không. Đây là một điểm rất thực dụng của paper: họ không bán một kiến trúc generator mới, mà bán một training-time regularizer có tác động lớn lên dynamics tối ưu.

Chuyển slide:  
Từ kiến trúc này, câu hỏi tiếp theo là tại sao phải align ở layer sớm.

## Slide 10 - Why early-layer alignment is enough
Mục tiêu: Giải thích kết quả ablation về độ sâu alignment.

Lời thoại:  
Kết quả ablation cho thấy align ở early blocks tốt hơn align ở sâu. Cách hiểu của tác giả là các block đầu nên học semantic scaffold, tức là khung ngữ nghĩa thô của vật thể và cảnh. Nếu ta ép alignment quá muộn, ta vô tình can thiệp vào phần mạng đang cần giải quyết chi tiết tinh hơn như texture, biên, hay cấu trúc cục bộ phục vụ generation. Nói cách khác, REPA mạnh nhất khi nó định hình representation sớm rồi rút lui, thay vì ép toàn bộ mạng luôn bám chặt vào feature ngoài. Đây là một insight quan trọng vì nó giải thích tại sao một regularizer nhìn có vẻ đơn giản lại không phá hỏng mục tiêu sinh ảnh, mà ngược lại còn hỗ trợ nó.

Chuyển slide:  
Sau khi biết gắn ở đâu, ta nhìn vào các thành phần nào thực sự quan trọng trong phương pháp.

## Slide 11 - What Table 2 says: the components that matter
Mục tiêu: Tóm tắt các ablation chính về target encoder và similarity.

Lời thoại:  
Table 2 cho thấy REPA khá bền vững trước nhiều lựa chọn thiết kế. Dùng encoder mạnh hơn thì FID và quality nhìn chung đều tốt hơn, nhưng không có nghĩa bắt buộc phải dùng encoder lớn nhất. DINOv2-B đã cho hiệu quả rất tốt, nghĩa là lợi ích không chỉ đến từ việc tăng parameter ở target model. Thêm nữa, cosine similarity đơn giản cũng đã cạnh tranh được với loss contrastive phức tạp hơn. Cách đọc quan trọng của slide này là: REPA không phải một mẹo mỏng manh chỉ chạy được ở một setting rất hẹp. Nó có vẻ nắm đúng một tín hiệu chung, đó là semantic alignment của hidden state.

Chuyển slide:  
Từ đây mình chuyển sang câu hỏi về scaling: phương pháp này còn hiệu quả khi model lớn hơn hay target encoder mạnh hơn không.

## Slide 12 - Scalability: stronger targets and larger DiTs help more
Mục tiêu: Trình bày xu hướng scalability của REPA.

Lời thoại:  
Hình ở slide này cho thấy ba xu hướng đi cùng nhau. Thứ nhất, target encoder càng mạnh thì feature supervision càng có chất lượng, và diffusion model được lợi nhiều hơn. Thứ hai, model nền càng lớn thì REPA càng phát huy rõ, nghĩa là lợi ích không bị mất đi khi scale up. Thứ ba, paper không quan sát một trade-off đơn giản kiểu discrimination tăng thì generation giảm. Thay vào đó, frontier giữa hai yếu tố này được đẩy đồng thời theo hướng tốt hơn. Đây là tín hiệu rất tích cực, vì nhiều phương pháp regularization thường chỉ cải thiện một phía rồi làm hại phía còn lại.

Chuyển slide:  
Bây giờ mình đi vào phần nổi bật nhất của paper: tốc độ hội tụ.

## Slide 13 - Convergence and efficiency
Mục tiêu: Nhấn mạnh speedup của REPA và ý nghĩa của đường FID theo training iteration.

Lời thoại:  
Đây là một trong những hình mạnh nhất của paper. Trục hoành là số bước train, trục tung là FID. Đường của REPA giảm nhanh hơn rõ rệt so với baseline SiT-XL/2. Ở nhiều mốc sớm, REPA đã chạm hoặc vượt chất lượng mà baseline phải mất rất lâu mới đạt được. Bảng bên phải tóm tắt lại thông điệp này bằng các mốc cụ thể: chẳng hạn REPA ở 400 nghìn bước đã ngang hoặc tốt hơn mức mà mô hình gốc cần hàng triệu bước. Vì vậy, REPA không chỉ cho final score tốt hơn, mà còn thay đổi hoàn toàn optimization trajectory. Đây là lý do paper nói việc train diffusion transformer "easier than you think".

Chuyển slide:  
Nếu chỉ có speedup thôi thì chưa đủ, nên slide sau sẽ đặt REPA vào bức tranh hệ thống rộng hơn trên ImageNet 256.

## Slide 14 - System-level comparison on ImageNet 256x256
Mục tiêu: Đặt REPA cạnh các baseline mạnh nhất.

Lời thoại:  
Ở bảng so sánh hệ thống này, ta thấy REPA không chỉ hơn vanilla SiT, mà còn cạnh tranh rất tốt với nhiều baseline mạnh khác trên ImageNet 256 nhân 256. Khi kết hợp thêm classifier-free guidance và guidance interval, paper báo cáo FID tốt nhất là 1.42. Quan trọng hơn, mức này đạt được với số epoch thấp hơn baseline SiT-XL/2. Điều đó làm lập luận của paper chắc hơn: lợi ích của representation alignment không chỉ xuất hiện ở setting trung gian hay metric phụ, mà đi tới cả kết quả hệ thống cuối cùng mà cộng đồng thường nhìn vào.

Chuyển slide:  
Sau số liệu định lượng, mình chuyển sang ví dụ định tính để xem khác biệt hiện ra như thế nào trong ảnh sinh.

## Slide 15 - Qualitative improvement appears early in training
Mục tiêu: Cho người nghe thấy hiệu quả trên ảnh sinh, không chỉ trên FID.

Lời thoại:  
Slide này minh họa một điểm rất trực quan: khi so cùng noise, cùng sampler và cùng số bước sample, ảnh từ model có REPA đã ổn định hơn sớm trong quá trình train. Vật thể nhìn rõ danh tính hơn, bố cục hợp lý hơn, và chi tiết không bị "bể" nhiều như baseline. Điều này rất ăn khớp với câu chuyện mà paper kể từ đầu: nếu semantic scaffold được học sớm hơn, quá trình generation về sau sẽ bớt lúng túng hơn. Dù hình qualitative luôn cần nhìn cẩn thận, nó ở đây đóng vai trò hỗ trợ rất tốt cho phần định lượng.

Chuyển slide:  
Slide kế tiếp sẽ gom các kết quả lại thành một diễn giải tổng quát hơn.

## Slide 16 - Why REPA helps both speed and final FID
Mục tiêu: Tổng hợp cơ chế hoạt động của REPA dưới góc nhìn thực nghiệm.

Lời thoại:  
Paper không đưa ra một định lý lý thuyết mạnh, nhưng họ đề xuất một câu chuyện cơ chế khá thuyết phục. Bước một, early hidden states hấp thụ semantic information từ encoder pretrained. Bước hai, vì semantics không còn phải học hoàn toàn ngầm từ denoising loss, bài toán tối ưu trở nên dễ hơn. Bước ba, các layer phía sau có thể dùng dung lượng mô hình cho phần thật sự generative, tức là hình học tinh, texture và chi tiết. Nói cách khác, REPA vừa đóng vai trò curriculum cho representation learning, vừa là một inductive bias hợp lý cho generator. Đây là lý do nó giúp cả tốc độ hội tụ lẫn final FID.

Chuyển slide:  
Tuy nhiên paper cũng có các giới hạn và câu hỏi mở, mình nói ngắn ở slide sau.

## Slide 17 - Limitations and future directions
Mục tiêu: Nêu điểm yếu và khoảng trống của paper.

Lời thoại:  
Có vài giới hạn đáng nhắc. Thứ nhất, alignment depth hiện vẫn chủ yếu là một lựa chọn thực nghiệm, chưa có giải thích nguyên lý thật chắc. Thứ hai, phần lớn kết quả của paper nằm ở latent image diffusion; các modality khác như video hay pixel-space diffusion vẫn còn mở. Thứ ba, vì dùng encoder pretrained ngoài, ta cũng phụ thuộc vào chất lượng và inductive bias của encoder đó. Cuối cùng, paper mới chỉ khám phá một phần của không gian thiết kế, ví dụ lịch thay đổi lambda theo thời gian hay target encoder động. Những hướng này có thể tạo ra các phiên bản mạnh hơn của REPA trong tương lai.

Chuyển slide:  
Mình kết lại phần chính bằng ba ý cần nhớ nhất.

## Slide 18 - What to remember
Mục tiêu: Chốt ba takeaways cho phần main talk.

Lời thoại:  
Nếu chỉ mang về ba ý, mình nghĩ nên mang ba ý này. Một là diffusion transformers thật ra đã học được feature có nghĩa, nhưng semantic alignment của chúng còn yếu và đắt đỏ để tối ưu. Hai là REPA biến vấn đề đó thành một auxiliary objective rất đơn giản: align hidden state nhiễu với clean pretrained features. Ba là khi representation learning tốt hơn, generation không chỉ đẹp hơn mà còn train nhanh hơn đáng kể. Với mình, đây là một paper tiêu biểu cho quan điểm rằng generative modeling và representation learning không nên bị tách rời. Chúng hỗ trợ nhau trực tiếp hơn ta thường nghĩ.

Chuyển slide:  
Từ đây nếu còn thời gian hoặc có câu hỏi, mình chuyển sang phần appendix.

## Slide 19 - Backup roadmap
Mục tiêu: Giới thiệu cấu trúc phần backup.

Lời thoại:  
Nếu bị hỏi sâu hơn, phần appendix của mình có ba nhóm nội dung. Nhóm một là phần nền toán học, gồm DDPM, stochastic interpolants, velocity và score. Nhóm hai là phần phân tích biểu diễn, như linear probing, CKNNA, và khác biệt giữa DiT với SiT. Nhóm ba là các kết quả mở rộng, gồm thêm ablation, chi tiết compute, ImageNet 512 và text-to-image. Tùy câu hỏi, mình sẽ nhảy tới slide phù hợp thay vì đi tuần tự.

Chuyển slide:  
Nếu có người hỏi về quan hệ giữa setup của paper và DDPM quen thuộc, mình dùng slide sau.

## Slide 20 - DDPM vs stochastic interpolants
Mục tiêu: So sánh hai cách nhìn về diffusion training.

Lời thoại:  
Nếu bị hỏi, mình sẽ giải thích rằng DDPM là cách nhìn rời rạc theo timestep, còn stochastic interpolants là cách nhìn liên tục giữa dữ liệu sạch và Gaussian noise. Trong DDPM, ta thường học dự đoán epsilon hoặc trực tiếp tham số hóa reverse transition. Trong stochastic interpolants hay flow matching, ta học một vector field hoặc score trên thời gian liên tục. REPA không phụ thuộc mạnh vào bên nào vì nó can thiệp ở hidden representation, không phụ thuộc trực tiếp vào sampler cụ thể. Đó là lý do paper có thể áp dụng ý tưởng tương tự cho cả SiT lẫn DiT.

Chuyển slide:  
Nếu câu hỏi đi vào công thức chi tiết hơn, mình dùng slide tiếp theo.

## Slide 21 - Velocity and score relations
Mục tiêu: Giải thích vì sao REPA không đụng đến phần lõi của diffusion math.

Lời thoại:  
Ở đây điều quan trọng là velocity và score chỉ là hai cách viết tương đương cho động lực học của quá trình sinh. Paper giữ nguyên toàn bộ phần này. Loss velocity vẫn là loss chính để học nghịch đảo corruption. Công thức score cũng vẫn được suy ra từ velocity như bình thường. Vậy nên khi nói REPA giúp training dễ hơn, không phải vì paper đã "hack" sampler hay đổi objective generative cốt lõi, mà vì họ bổ sung semantic supervision đúng chỗ trong hidden state.

Chuyển slide:  
Nếu có người hỏi các metric biểu diễn trên slide 4 nghĩa là gì, mình chuyển sang slide sau.

## Slide 22 - What are CKNNA and linear probing measuring?
Mục tiêu: Làm rõ ý nghĩa của hai metric phân tích biểu diễn.

Lời thoại:  
Linear probing là cách quen thuộc: đóng băng feature rồi train một lớp tuyến tính để xem class semantics có tách ra tốt không. Nó cho biết feature có tuyến tính hóa được thông tin phân loại hay không. CKNNA là một metric alignment mềm hơn CKA, tập trung vào cấu trúc lân cận gần nhất giữa hai không gian biểu diễn. Vì so sánh giữa diffusion feature và SSL feature là so sánh giữa hai họ mô hình rất khác nhau, CKNNA phù hợp hơn một metric toàn cục quá cứng. Trong paper, probing trả lời câu hỏi "feature có semantic không", còn CKNNA trả lời câu hỏi "feature có giống những semantic feature mạnh mà ta tin cậy hay không".

Chuyển slide:  
Nếu người nghe muốn nhìn rõ hơn block mà REPA can thiệp, mình dùng slide sau.

## Slide 23 - DiT / SiT block details
Mục tiêu: Giải thích chi tiết block transformer dùng trong backbone.

Lời thoại:  
Slide này cho thấy DiT và SiT về cơ bản là transformer chạy trên patch token của latent image. Timestep và điều kiện đi vào thông qua cơ chế modulation kiểu AdaLN-zero. REPA lấy ra hidden state ở một trong các block đó rồi đưa sang projector MLP để align với target encoder. Điều cần nhấn mạnh là projector không thay thế block gốc và cũng không can thiệp vào dòng suy luận chính khi sampling. Nó chỉ là một nhánh phụ để huấn luyện representation tốt hơn.

Chuyển slide:  
Một câu hỏi khác hay gặp là liệu lập luận này chỉ đúng với SiT hay không.

## Slide 24 - The same behavior also appears in DiT
Mục tiêu: Chỉ ra hiện tượng semantic gap cũng tồn tại trong DiT.

Lời thoại:  
Nếu chỉ thấy kết quả trên SiT thì có thể nghi ngờ đây là hiện tượng riêng của flow matching. Nhưng slide này cho thấy DiT cũng có semantic gap và alignment gap tương tự khi so với DINOv2. Nghĩa là vấn đề paper chỉ ra có vẻ mang tính họ mô hình diffusion transformer rộng hơn, chứ không phải artifact của một biến thể riêng. Điều này làm REPA trở nên hấp dẫn hơn như một ý tưởng tổng quát: nếu điểm nghẽn là internal representation, thì nhiều diffusion transformer khác nhau có thể cùng hưởng lợi.

Chuyển slide:  
Nếu có câu hỏi về độ nhạy của phương pháp theo timestep, target encoder hay lambda, mình dùng slide sau.

## Slide 25 - More ablations: timesteps, target encoders, and \(\lambda\)
Mục tiêu: Chuẩn bị backup cho câu hỏi về robustness của REPA.

Lời thoại:  
Phần ablation bổ sung này cho thấy REPA không chỉ có tác dụng ở một mức noise duy nhất. Khoảng cách biểu diễn được cải thiện trên nhiều timestep khác nhau. Ngoài DINOv2, các target encoder khác như MAE hay MoCov3 cũng giúp, dù mức độ hiệu quả có khác nhau. Về trọng số lambda, kết quả cho thấy có một vùng tương đối ổn định sau khi lambda đủ lớn để semantic supervision thật sự có trọng lượng. Nếu bị hỏi "phương pháp này có quá nhạy siêu tham số không", câu trả lời của paper nhìn chung là không quá nhạy.

Chuyển slide:  
Nếu người nghe quan tâm về chi phí triển khai, slide tiếp theo tóm tắt phần đó.

## Slide 26 - Hyperparameters and compute details
Mục tiêu: Trả lời câu hỏi về compute overhead và setup huấn luyện.

Lời thoại:  
Về triển khai, paper giữ hầu hết setup huấn luyện như backbone gốc. Optimizer vẫn là AdamW, batch size và sampler giữ ở cấu hình chuẩn của họ. Phần thêm vào chủ yếu là chạy encoder pretrained và projector MLP. Tác giả cũng lưu ý có thể precompute feature của encoder ngoài để giảm chi phí huấn luyện. Vì vậy, overhead của REPA là có thật nhưng tương đối dễ quản lý, đặc biệt nếu so với lợi ích lớn về số bước train tiết kiệm được.

Chuyển slide:  
Nếu cần chi tiết số liệu hơn nữa, mình có bảng backup tiếp theo.

## Slide 27 - Detailed quantitative results
Mục tiêu: Có số liệu backup cho các mốc FID và cấu hình guidance.

Lời thoại:  
Slide này hữu ích khi người nghe muốn biết các mốc cụ thể thay vì chỉ nhìn xu hướng. Ta thấy cùng một backbone, REPA cải thiện liên tục từ mốc 400 nghìn bước tới 4 triệu bước. Phần bảng bên phải nhấn mạnh thêm rằng guidance interval cộng với REPA tạo ra kết quả mạnh nhất trên ImageNet 256. Nếu bị hỏi "REPA mạnh nhất ở giai đoạn đầu train hay mạnh cả ở cuối", dữ liệu ở đây cho thấy cả hai, nhưng hiệu ứng ấn tượng nhất chính là giai đoạn đầu và trung bình.

Chuyển slide:  
Tiếp theo là phần mở rộng lên độ phân giải cao hơn.

## Slide 28 - Extension to ImageNet 512x512
Mục tiêu: Chứng minh REPA không bị khóa ở ImageNet 256.

Lời thoại:  
Nếu còn thời gian, mình sẽ nói ngắn rằng paper còn kiểm tra trên ImageNet 512 nhân 512. Kết quả qualitative ở đây cho thấy ảnh sinh vẫn khá mạnh ở độ phân giải lớn hơn. Quan trọng hơn, tác giả báo cáo rằng REPA vượt baseline SiT-XL/2 với số iteration ít hơn nhiều, tức là xu hướng speedup vẫn giữ. Nói cách khác, lợi ích của representation alignment không biến mất khi bài toán trở nên khó hơn về mặt độ phân giải.

Chuyển slide:  
Một mở rộng còn thú vị hơn là chuyển từ unconditional hoặc class-conditional image generation sang text-to-image.

## Slide 29 - Text-to-image extension
Mục tiêu: Trình bày khả năng mở rộng sang setting có text conditioning.

Lời thoại:  
Ở phần này paper áp dụng ý tưởng REPA lên MMDiT cho text-to-image trên MS-COCO. Đây là một test khá quan trọng, vì trong setting text-to-image, model còn phải hòa trộn thông tin từ text embeddings chứ không chỉ học visual semantics. Kết quả của paper cho thấy FID vẫn được cải thiện rõ ở cả ODE lẫn SDE setup. Điều đó gợi ý rằng alignment với strong visual representation vẫn hữu ích ngay cả khi pipeline có thêm text conditioning. Nếu bị hỏi "REPA có chỉ là mẹo cho class-conditional ImageNet không", slide này là câu trả lời phủ định khá mạnh.

Chuyển slide:  
Slide cuối cùng trong appendix sẽ cho trực giác trực quan hơn về hidden feature map.

## Slide 30 - Feature maps and extra qualitative samples
Mục tiêu: Kết thúc backup bằng bằng chứng trực quan ở mức representation.

Lời thoại:  
Đây là slide mình khá thích vì nó kết nối trực giác representation với ảnh sinh cuối cùng. Bên trái là PCA visualization của feature maps theo layer ở nhiều mức noise khác nhau. Ta thấy với REPA, các feature map có xu hướng coarse-to-fine rõ hơn, nghĩa là lớp đầu mang cấu trúc lớn, còn lớp sau tinh chỉnh chi tiết. Baseline thì noisy hơn và kém ổn định hơn, đặc biệt ở noise lớn. Bên phải là một số qualitative sample bổ sung, nhắc lại rằng tín hiệu tốt hơn trong hidden space thực sự đi ra ngoài thành ảnh sinh tốt hơn. Nếu kết luận một câu, thì REPA thuyết phục không chỉ ở FID, mà còn ở cách nó làm internal representations trở nên có tổ chức hơn.

Chuyển slide:  
Hết phần backup. Nếu còn câu hỏi, mình sẽ quay lại slide tương ứng với chủ đề người nghe quan tâm nhất.
