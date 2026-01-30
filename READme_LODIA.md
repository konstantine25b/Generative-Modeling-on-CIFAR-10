# Paper 1: Generative Modeling by Estimating Gradients of the Data Distribution

Paper ამბობს: არ ვასწავლით model-ს პირდაპირ p(x)-ს ან likelihood-ს. ამის ნაცვლად ვასწავლით score function-ს:

∇xlogp(x)

და მერე Langevin dynamics-ით (MCMC) ვაგენერირებთ სემპლებს. მაგრამ “რეალურ” მონაცემებზე score ხშირად ill-defined არის (manifold hypothesis), ამიტომ ისინი ამატებენ სხვადასხვა დონეზე Gaussian noise-ს, ასწავლიან Noise Conditional Score Network (NCSN)-ს და sampling-ს აკეთებენ annealed Langevin dynamics-ით.

თუ ვიცით score ∇xlogp(x), მაშინ Langevin dynamics შეგვიძლია გამოვიყენოთ და მივიღოთ სემპლები. ამიტომ ისინი პირდაპირ score-ს ასწავლიან score matching-ით, მერე sampling-ს აკეთებენ. 

დაბრკოლება: 
1) Manifold hypothesis → score შეიძლება undefined იყოს ambient space-ში.

2) Low-density regions → score estimation ცუდია იქ, სადაც data თითქმის არ არის, ხოლო sampling ხშირად იწყება სწორედ “შორს” (noise-დან), ანუ ცუდ რეგიონში. 

რა არის score matching + Langevin dynamics:

1.Score matching = score estimation
Score network  𝑠𝜃(𝑥) უნდა მიუახლოვდეს ∇xlogPdata(x).

2.Denoising Score Matching (DSM)
DSM იდეა: ჯერ “დააზიანე” x noise-ით და მერე ასწავლე perturbed distribution-ის score. ანუ network სწავლობს “როგორ უნდა დააბრუნოს” noisy sample უკან density-ში. 

3.Sliced Score Matching (SSM)
SSM იყენებს random projection vector v-ს და approximate-ს აკეთებს trace-term-ის:
უპირატესობა: original distribution-ის score-ს სწავლობს; მინუსი: forward-mode autodiff-ის გამო ~4x მძიმეა.

4.Sampling with Langevin dynamics
ჩვენი score network სწავლობს “სად უნდა წავიდეს სურათი”, ანუ მოცემულ სურათზე/ნარევზე გვაძლევს მიმართულებას, რომელიც მას უფრო მაღალი probability-ის (უფრო “realistic”) რეგიონში წაიყვანს. Sampling-ის დროს ჩვენ ვიწყებთ შემთხვევითი noise-ით და ბევრჯერ ვიმეორებთ პატარა განახლებებს, რომ ნელ-ნელა მივიღოთ რეალისტური სურათი.

Basic Langevin step
Sampling-ში თითო ნაბიჯზე ხდება ორი რამ:

1.Drift / denoise push
network-ის “direction” ვუმატებთ სურათს, რომ ის რეალური მონაცემების მიმართულებით გადაადგილდეს.

2.Random noise injection
ასევე ვუმატებთ პატარა შემთხვევით noise-ს, რომ პროცესი არ “ჩაეჭედოს” და შეძლოს სხვადასხვა mode-ებს შორის მოძრაობა (diversity და mixing).

ეს ორი კომპონენტი ერთად ქმნის “controlled random walk”-ს:

direction გვქაჩავს data-სკენ
randomness გვაძლევს exploration-ს


Noise Conditional Score Networks (NCSN)!

NCSN არის ერთი neural network, რომელიც იღებს ორ რამეს:

1.სურათს (რომელიც შეიძლება იყოს noise-ით დაზიანებული)
2.noise level (რამდენად noisy არის)
და აბრუნებს score-ს — მიმართულებას თითო პიქსელზე, თუ როგორ უნდა “გასწორდეს” ეს სურათი.

არქიტექტურა:
რადგან network-ის output არის image-sized field (თითო პიქსელზე მიმართულება), გამოიყენება dense prediction სტილის არქიტექტურები:

U-Net / RefineNet სტილის skip connections
dilated convolutions (დიდი receptive field, resolution-ის დაკარგვის გარეშე)
normalization ფენები, რომლებიც conditioning-ს აკეთებენ noise level-ზე

ტრენინგი: 
Training-ის დროს ვაკეთებთ მარტივ პროცესს:
2.ვიღებთ ნამდვილ სურათს.
3.ვუმატებთ Gaussian noise-ს გარკვეული დონით.
4.ახლა გვაქვს “noisy image”.
5.ვიცით ზუსტად რა noise დავუმატეთ → ვიცით, რა მიმართულებით უნდა “დაბრუნდეს” სურათი.
6.network-ს ვასწავლით ამ მიმართულების გამოცნობას.

სემპლინგი:
Training-ის მერე network-ს შეუძლია თქვას:
“ამ noisy image-ს რომელი მიმართულებით გადავწიოთ, რომ უფრო რეალისტური გახდეს?”

Sampling კეთდება Annealed Langevin Dynamics-ით.

ნაბიჯები:
1.ვიწყებთ pure random noise-ით.
2.გვაქვს noise levels დიდი → პატარა.

3.თითო დონეზე:

რამდენიმე Langevin step,
ყოველ step-ზე:,
network-ის მიმართულებით “ვაწევთ” სურათს რეალური მონაცემებისკენ.

4.ვუმატებთ პატარა შემთხვევით noise-ს (exploration-ისთვის)

5.noise დონე ნელ-ნელა მცირდება.
ბოლოს ვიღებთ რეალისტურ სურათს.

--
რატომ არის step size დაკავშირებული noise level-თან?

Sampling-ის დროს ნაბიჯის ზომა იზრდება noise level-თან ერთად.
მიზანია, რომ signal-to-noise ratio იყოს სტაბილური ყველა დონეზე.

თუ step size ძალიან დიდია პატარა noise-ზე → instability
თუ ძალიან პატარაა დიდ noise-ზე → ნელი mixing

---
რატომ სჭირდება annealing (noise schedule)?

თუ პირდაპირ პატარა noise-ზე დავიწყებთ:
1.chain შეიძლება ცუდ რეგიონში გაიჭედოს
2.ვერ გადავა mode-ებს შორის

დიდ noise-ზე დაწყება გვაძლევს:
2.უკეთეს global mixing-ს
3.სწორ mode proportions-ს
-----------------------------------------------------------------------------

Paper 2: Improved Techniques for Training Score-Based Generative Models
ქაღალდის მიზანია score-based generative modeling (NCSN + annealed Langevin dynamics) გადაიყვანოს მაღალ რეზოლუციებზე (64×64…256×256) და გახადოს სტაბილური, რადგან “ძველი” კონფიგურაციები (noise scales, sampling hyperparams) კარგად მუშაობდა ძირითადად 32×32-ზე.

დიდი განსხვავება პირველ paper-თან: პირველი paper პრაქტიკულად “introduces the framework + works on CIFAR-10 32×32”, ხოლო ეს paper არის “engineering + theory-driven tuning guide” რომ იმუშაოს high-res-ზე

რატომ ფეილდება/სუსტდება NCSN როცა D (dimensionality) იზრდება:

1.noise scales: ძველი რეკომენდაციები არის “არაგამჭვირვალე” და high-res-ზე ცუდია. 
2.sampling via Langevin: high-dim-ში Langevin trajectories მარტივად გადიან “no data” რეგიონებში და ვერ კონვერგირდება, მით უმეტეს როცა score imperfect-ია.


1.Choosing the initial noise scale (the largest noise level)
როგორ იყო პირველ paper-ში (2019):
პირველ paper-ში noise levels-ის ყველაზე დიდი მნიშვნელობა ხშირად აირჩეოდა როგორც პრაქტიკული კონსტანტა (მაგ. “საკმარისად დიდი”), და CIFAR-10-ზე ეს მუშაობდა. აქცენტი იყო იმაზე, რომ “დიდ noise-ზე mixing ადვილია, პატარა noise-ზე დეტალები იხვეწება”, მაგრამ არ იყო მკაფიო წესი როგორ ავირჩიოთ რამდენად “დიდი” უნდა იყოს დასაწყისი.

რა დაამატა/შეცვალა NCSNv2 (2020):
NCSNv2 ამბობს: თუ initial noise ძალიან პატარაა, sampler პრაქტიკულად “დარჩება” ერთ მონაცემთან/ერთ mode-თან ახლოს და ვერ “გადახტება” სხვა რეგიონებში (mixing fails). ეს problem განსაკუთრებით მძაფრდება მაღალი განზომილებებისა და მაღალი რეზოლუციის სურათებზე.

ამიტომ ისინი გვთავაზობენ უფრო principled არჩევას: initial noise უნდა იყოს დაკავშირებული dataset-ის “მასშტაბთან/დიამეტრთან” (ანუ რამდენად დაშორებულია ერთმანეთისგან მონაცემები), და არა უბრალოდ ერთ-ორი fixed რიცხვი ყველა dataset-ზე. პრაქტიკაში ეს ნიშნავს: “დასაწყისის noise ისეთი დიდი უნდა იყოს, რომ dataset-ის სხვადასხვა ნაწილებს შორის მოძრაობა რეალურად შესაძლებელი იყოს”.


2.Picking intermediate noise levels (why geometric schedules appear)
როგორ იყო პირველ paper-ში (2019):
პირველ paper-ში გამოიყენებოდა რამდენიმე noise level და მათ შორის გადასვლა პრაქტიკულად მუშაობდა (ხშირად geometric ტიპის grid-ით). მაგრამ rationale იყო ძირითადად ემპირიული: “ბევრ დონეზე სწავლება საჭიროა, რომ sampling-მა იმუშაოს” და “annealing ეხმარება mode-mixing-ს”. არ იყო მკაფიო ახსნა, რატომ უნდა იყოს დონეები ასე განაწილებული და რა ხდება თუ “არასწორად” განაწილდება.

რა დაამატა/შეცვალა NCSNv2 (2020):
NCSNv2 პირდაპირ ამბობს: მთავარი მოთხოვნაა, რომ მეზობელ noise დონეებს შორის გადასვლისას “მაღალი probability რეგიონები ერთმანეთს ფარავდეს” — თორემ chain “გაწყდება”: ერთი დონიდან მეორეზე გადასვლისას sample აღმოჩნდება ისეთ სივრცეში, სადაც model-ს ცუდი guidance აქვს.

ამ reasoning-ით ისინი ამართლებენ ისეთ schedule-ს, სადაც მეზობელ დონეებს შორის “გადაფარვა” დაახლოებით მუდმივია. ამის შედეგად ხშირად გამოდის geometric მსგავსი განაწილება და კიდევ ერთი პრაქტიკული დასკვნა: low-res dataset-ზე (მაგ. 32×32) 10 დონე შეიძლება მუშაობდეს, მაგრამ უფრო რთულ/high-res setting-ში იგივე რაოდენობა ხშირად არასაკმარისია და საჭიროა მეტი დონე ან უკეთესი განაწილება.



3.Noise conditioning without per-level parameters (memory-efficient conditioning)
როგორ იყო პირველ paper-ში (2019):
პირველ paper-ში noise conditioning ხშირად იყო “architectural”: normalization ფენებში (მაგ. conditional normalization) თითო noise level-ს ჰქონდა თავისი learned scale/bias პარამეტრები. ეს მუშაობდა, მაგრამ როცა noise levels იზრდება (L დიდია), მოდელის პარამეტრებიც იზრდება და ეს approach ნაკლებად მოქნილი ხდება.

რა დაამატა/შეცვალა NCSNv2 (2020):
NCSNv2 ამჩნევს რეგულარობას: score-ის “ზომა” ბუნებრივად იცვლება noise level-ის მიხედვით (დიდ noise-ზე score პატარაა, პატარა noise-ზე დიდი). აქედან მოჰყავს უფრო მარტივი და ზოგადი conditioning იდეა: network-ის output-ს შეიძლება ჰქონდეს “სკეილინგი” noise level-ით ისე, რომ აღარ დაგვჭირდეს ცალკე learned პარამეტრები ყველა დონეზე.

პრაქტიკული მოგება:
ნაკლები memory და პარამეტრები
უფრო ადვილია ბევრი noise level-ის გამოყენება
უფრო მარტივი ხდება continuous noise/time conditioning-ის მხარდაჭერა

4.Tuning sampling hyperparameters to match the noise schedule
პირველ paper-ში sampling-სთვის ხშირად გამოიყენებოდა ერთი ფიქსირებული “recipe”: რამდენი Langevin step თითო noise level-ზე და დაახლოებით რა step size. ასევე იყო step size-ის scaling idea noise level-ის მიხედვით (ინტუიცია: რომ sampling “სტაბილური” ყოფილიყო სხვადასხვა დონეზე). მაგრამ ეს პარამეტრები დიდწილად იყო ერთი კონკრეტული setting-ისთვის შერჩეული და ზოგ dataset-ზე/რეზოლუციაზე არ გადაიტანებოდა.

NCSNv2 ამბობს: sampling hyperparams (რამდენი ნაბიჯი, რა step size) უნდა აირჩეს noise schedule-თან შეთანხმებით. ანუ თუ შეცვალე:

1.initial noise
2.დონეების რაოდენობა
3.დონეების განაწილება

მაშინ sampling-ის ნაბიჯების რაოდენობა და step size აღარ შეიძლება “იგივე დარჩეს”. წინააღმდეგ შემთხვევაში შეიძლება მიიღო:

1.ძალიან ნელი mixing (თუ ნაბიჯები/step size პატარაა)
2.instability / artifacts (თუ ნაბიჯები/step size დიდია)

ეს არის ზუსტად ის ადგილი, სადაც ჩვენს პროექტში გაჩნდება “SNR-based” tuning იდეა: sampling-ის კონტროლი ისე, რომ სხვადასხვა დონეზე მოძრაობა იყოს თანმიმდევრული.

5.EMA of weights for more stable and cleaner samples
როგორ იყო პირველ paper-ში (2019):
პირველ paper-ში ყურადღება იყო score learning + annealed Langevin sampling-ზე და არქიტექტურულ ნაწილზე. EMA (Exponential Moving Average) როგორც sampling-time best practice არ იყო მთავარი კომპონენტი.

რა დაამატა/შეცვალა NCSNv2 (2020):
NCSNv2 introduces EMA როგორც პრაქტიკული “სტაბილიზატორი”:

1.training-ის დროს ვაგროვებთ weights-ის მოძრავ საშუალოს
2.sampling-ისას ვიყენებთ EMA weights-ს (არა raw weights-ს)

პრაქტიკული ეფექტი: ნაკლები “ფერის აცდენა”, ნაკლები არტეფაქტები და ზოგადად უფრო სტაბილური ხარისხი checkpoint-ებს შორის. “ჩვენ sampling-ს ვაკეთებთ EMA weights-ით, რადგან ეს ამცირებს ვიზუალურ არტეფაქტებს და ზრდის სტაბილურობას”. 

# ექსპერიმენტი N1
## Noise Level Count Ablation
------
Time embedding არის noise level-ის ჩასმა ქსელში. sigma ანუ noise scale  გადაგვყავს ვექტორში რომელსაც რესბლოკებში ვუმატებთ.ვქმნით ახალ ლეიერს, გადავცემთ dim-ს ანუ ვექტორის ზომას(რამდენი ფიჩერი ექნება). 
nn.Linear(in_features, out_features) = fully connected layer.
fc1 იღებს 1 რიცხვს ანუ სიგმას და აკეთებს დიმ ზომის ვექტორს. fc2 კიდევ ერთი  fully connected რომ ვექტორი უკეთესი და ძიერი გამოვიდეს.

forward(self,t) - t არის სიგმა.
view(-1,1) ნიშნავს - გადაალაგე ტენსორ ისე რომ ფორმა ქონდეს (batch,1).
-1 ნიშნავს რომ პაითორჩი მისით დათვლის batch sizes.
რატომ ვშვებით ამას, იმიტომ რომ nn.Linear  ელოდება [batch,features] ფორმას.
h = F.relu(self.fc1(t)) - თავიდან ქმნის [batch,dim] ვექტორებს. რელუ არაწრფივს ხდის(უარყოფითი -> 0). ემბედინგს ვაქცევთ არაწრფივს რათა სიგმას ეფექტი არ იყოს მხოლოდ ხაზოვანი.
return F.relu(self.fc2(h)) - fc2 ისევ გარდაქმნის ემბედინგს და ისევ რელუს მოსდებს. შედეგი: [batch,dim]  embeding  ვექტორი, რომელიც გამოიყენება ქონდიშენად.
-------
ResBlock - Residual Block conditioning
ResBlock  არის ბლოკი, რომელიც აკეთებს Conv->Norm->ReLU, ამატებს სიგმა ემბედინგს, ისევ Conv->Norm  დბოლოს აუთფუთი აქვს F(x) + x.

ResBlock(nn.Module): - გადაეცემა input feature maps channels, რამდენ channel-ს გამოვიტანთ ამ block-იდან, embedding vector-ის ზომა (TimeEmbedding-ის output dim).

Conv2d(in_channels, out_channels, kernel_size, padding) - output: [batch, out_ch, H, W]
self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1) - მეორე convolution იგივე out_ch→out_ch. block-ის “მთავარი ტრანსფორმაცია” ხდება ორი conv-ით.
self.emb_proj = nn.Linear(emb_dim, out_ch) - ემბედინგ ვექტორი [batch, emb_dim] უნდა გადაიქცეს [batch, out_ch]-ად.
ეს არის კონდიშენინგ მექანიზმი: სიგმა აწესებს შიფტს ფიჩერებზე.

self.gn1 = nn.GroupNorm(8, out_ch)
self.gn2 = nn.GroupNorm(8, out_ch)
არგუმენტები: GroupNorm(num_groups, num_channels)
8 ჯგუფი ნიშნავს: out_ch channel-ები იყოფა 8 ჯგუფად და normalize ხდება ჯგუფებად
GroupNorm კარგია diffusion/score მოდელებში, რადგან BatchNorm ხშირად ცუდად მუშაობს

self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch!=out_ch else nn.Identity()
residual connectionში ინფუთი იქსი უნდა დავუმატოთ აუთფუთს, მაგრამ თუ in_ch != out_ch მაშინ შეიფები არ ემთხვევა, ამიტო ვიყენებთ 1x1  convolution-ს პროექციისთვის: [in_ch] → [out_ch]. თუ channels ემთხვევა, skip უბრალოდ identity.

forward(self,x,emb): გადაეცემა იქსი ანუ feature map [batch, in_ch, H, W], sigma embedding [batch, emb_dim].
h = F.relu(self.gn1(self.conv1(x))) - self.conv1(x) -> convolution: [batch, out_ch, H, W], self.gn1(...) → normalize features (სტაბილურობა), F.relu(...) → nonlinearity.
h = h + self.emb_proj(emb).view(emb.size(0), -1, 1, 1) - self.emb_proj(emb) → [batch, out_ch], .view(emb.size(0), -1, 1, 1) → reshape to [batch, out_ch, 1, 1], ანუ თითო channel-ზე ერთი scalar bias/addition.
მერე h + ...: - [batch, out_ch, 1, 1] ემატება [batch, out_ch, H, W]-ს, შედეგი: sigma გავლენას ახდენს ყველა პიქსელზე (H,W) ერთნაირად, მაგრამ განსხვავებულ channel-ებზე განსხვავებულად.
h = self.gn2(self.conv2(h)) მეორე conv + GroupNorm (ამ ხაზში ReLU ჯერ არ გვაქვს).
return F.relu(h + self.skip(x)) - self.skip(x) არის residual branch (identity ან 1×1 conv), h + skip(x) = residual sum, ReLU ბოლოს, output activation. საბოლოო output shape: [batch, out_ch, H, W]
----------
UNetScore — U-Net structure score prediction-ისთვის
Score model output უნდა იყოს იგივე ზომის tensor, რაც input image:
input: [batch, 3, 32, 32]
output: [batch, 3, 32, 32]
ეს output არის “vector field”:
თითო პიქსელზე 3 კომპონენტი (RGB მიმართულება), როგორ “გადავწიოთ” sample data-სკენ.

UNetScore(nn.Module): - base_ch არის პირველი stage-ის channel count (64).
elf.time_emb = TimeEmbedding(128) - sigma embedding vector size = 128.
self.down1 = ResBlock(3, base_ch, 128)
self.down2 = ResBlock(base_ch, base_ch*2, 128)
self.down3 = ResBlock(base_ch*2, base_ch*4, 128)
ჩენელები იზრდება, 3->64->128->256

self.mid = ResBlock(base_ch*4, base_ch*4, 128) - ყველაზე დაბალ resolution-ზე feature processing.

self.up3 = ResBlock(base_ch*8, base_ch*2, 128)
self.up2 = ResBlock(base_ch*4, base_ch, 128)
self.up1 = ResBlock(base_ch*2, base_ch, 128)
რატომ ასეთი input channels? U-Net-ში decoding დროს ვაკეთებთ concat([upsampled, skip]), ამიტომ channels ემატება.
მაგალითად up3: upsampled mid: base_ch*4 (256), skip from d3: base_ch*4 (256), concat → base_ch*8 (512), output ვაბრუნებთ base_ch*2 (128)

self.out = nn.Conv2d(base_ch, 3, 3, padding=1)
final layer: 64→3
padding=1 რომ output ისევ 32×32 იყოს.
ეს არის score estimate.

self.pool = nn.AvgPool2d(2)
downsampling: H,W ნახევრდება (32→16→8→4)
AvgPool2d(2) ნიშნავს 2×2 window average.

self.up = nn.Upsample(scale_factor=2)
upsampling: H,W ორმაგდება (4→8→16→32)
earest interpolation default-ია


forward(self, x, sigma): - x: input image or noisy image [batch,3,32,32], sigma: [batch].
emb = self.time_emb(torch.log(sigma)) - sigma-ს იღებენ log-scale-ზე, time_emb returns [batch,128]
d1 = self.down1(x, emb) - d1 shape: [batch,64,32,32]
d2 = self.down2(self.pool(d1), emb) - pool(d1) → [batch,64,16,16], down2 → [batch,128,16,16]
d3 = self.down3(self.pool(d2), emb) pool(d2) → [batch,128,8,8], down3 → [batch,256,8,8]
mid = self.mid(self.pool(d3), emb) pool(d3) → [batch,256,4,4], mid → [batch,256,4,4]
u3 = self.up3(torch.cat([self.up(mid), d3], 1), emb) - self.up(mid) → upsample mid: [batch,256,8,8], 
torch.cat([...], 1) → concatenate on channel dim (dim=1) means [batch,256,8,8] + [batch,256,8,8] → [batch,512,8,8]. up3 ResBlock: input 512 → output 128: [batch,128,8,8]
u2 = self.up2(torch.cat([self.up(u3), d2], 1), emb) - up(u3) → [batch,128,16,16], concat with d2 [batch,128,16,16] → [batch,256,16,16], up2 outputs [batch,64,16,16].
u1 = self.up1(torch.cat([self.up(u2), d1], 1), emb) - up(u2) → [batch,64,32,32], concat with d1 [batch,64,32,32] → [batch,128,32,32], up1 outputs [batch,64,32,32].

return self.out(u1) - final conv: [batch,64,32,32] → [batch,3,32,32], ეს არის predicted score field.
------------



# ექსპერიმენტი N2

## NCSN U-Net - Noise schedule ablation (geometric vs linear vs SNR-based). 

მოკლე მიმოხილვა ჩვენი ცვლადების:

1) Geometric Noise Schedulე
noise levels (σ values) ეცემა მუდმივი ფარდობითი კოეფიციენტით.
ანუ ყოველი მომდევნო σ დაახლოებით არის წინა σ-ის მრავლობითი შემცირება.

ეს ნიშნავს: noise დონეები განლაგებულია log-scale-ზე, არა linear scale-ზე.

2) Linear Noise Schedule
Linear schedule-ში σ values ეცემა მუდმივი სხვაობით.
σ: 50 → 40 → 30 → 20 → 10 → ...

პრობლემა:
Linear schedule-ში ხდება: large distribution jump → score guidance invalid → random walk

Linear spacing იწვევს:
1.high σ რეგიონში distributions ძალიან შორსაა ერთმანეთისგან
2.overlap თითქმის არ არსებობს

რა ხდება sampling დროს:

პირველი რამდენიმე დონე: network ვერ პოულობს data direction-ს,
noise dominates,
chain ვერ "იჭერს" manifold-ს.

3) SNR-based Step Size (Geometric σ + tuned α)
Noise levels იგივეა geometric schedule-ის, მაგრამ Langevin step size (α) ირჩევა ისე, რომ:
signal-to-noise ratio იყოს დაბალანსებული.

ანუ update step-ში:
score contribution,
injected noise



იდეა:

ამ პროექტში ვსწავლობთ score-based generative modeling-ს CIFAR-10 dataset-ზე Noise Conditional Score Network (NCSN) არქიტექტურის გამოყენებით.

ჩვენ არ ვიყენებთ labels-ს.
მიზანია პირდაპირ ვისწავლოთ სურათების probability density structure — ანუ როგორია “რეალური სურათების სივრცე”.

მოდელი სწავლობს არა პირდაპირ სურათების გენერაციას, არამედ:

რომელი მიმართულებით უნდა “გავწიოთ” noisy image, რათა მივუახლოვდეთ data manifold-ს.

ეს მიდგომა fundamentally განსხვავდება GAN-ებისგან და VAE-ებისგან.


არქიტექტურა:
Score model აფასებს:
“noise-ით დაბინძურებული სურათის შემთხვევაში, სად არის data manifold?”
ანუ network-ის output არის vector field იმავე ზომის რაც სურათი.

რატომ U-Net?

Score field უნდა იყოს:
ლოკალურად ზუსტი (texture, edge, color)
გლობალურად თანმიმდევრული (object-like structure)
U-Net ამ ორს აერთიანებს.

Network სტრუქტურა:
Input:
3×32×32 image
noise level (sigma embedding)

Encoder (Down path)
Conv → ResBlock → Downsample
როლი: უფრო დიდი receptive field → global context

Bottleneck
ყველაზე კომპაქტური representation
აქ network ხედავს image-ს “სრული კონტექსტით”

Decoder (Up path)
Upsample → ResBlock
skip connections encoder-დან

Output Layer
Conv layer → score estimate (3×32×32)


Residual Blocks

ResBlocks საშუალებას აძლევს:
deep network training
gradient flow stabilization
better feature reuse

Noise Conditioning

Network უნდა იცოდეს noise level.
sigma → embedding → injected in blocks.

მნიშვნელობა:
Sigma დიდი	         Sigma პატარა
Structure learning	Detail refinement


ტრენინგი:
პროცესი:

clean image,
random noise level,
add Gaussian noise,
train network to predict denoising direction,
მოდელი სწავლობს multi-scale gradient field.

სემპლინგი:
ჩვენ ვიყენებთ annealed Langevin dynamics:

random start
step-by-step refinement
noise level decreases


| Schedule  | FID        |
| --------- | ---------- |
| Geometric | **231.21** |
| Linear    | **477.47** |
| SNR-based | **230.75** |
