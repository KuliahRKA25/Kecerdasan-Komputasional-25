#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.2cm, left: 2.5cm, right: 2.5cm),
  numbering: "1",
  number-align: right + bottom,
)

#set text(
  font: ("Libertinus Serif", "Liberation Serif"),
  size: 11pt,
  lang: "id",
)

#set text(font: "New Computer Modern", size: 11pt, lang: "id")
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1.")
#set list(indent: 1.2em, body-indent: 0.5em, spacing: 0.4em)

// ---------- Komponen ----------
#let placeholder(text: "Isi bagian ini.") = [
  #text(fill: rgb("666666"), style: "italic")[#text]
]

#let unnumbered-title(body) = {
  set par(first-line-indent: 0pt)
  align(center, text(weight: "bold", size: 11pt, body))
}

#let figure-box(caption: "Judul gambar", height: 6cm) = figure(
  rect(
    width: 100%,
    height: height,
    stroke: 0.8pt + rgb("999999"),
    inset: 12pt,
    align(center + horizon, placeholder("Tempatkan gambar di sini")),
  ),
  caption: caption,
  kind: image,
)

#let empty-table(
  caption: "Judul tabel",
  columns: 4,
  headers: ("Kolom 1", "Kolom 2", "Kolom 3", "Kolom 4"),
  rows: 3,
) = figure(
  kind: table,
  caption: caption,
  table(
    columns: columns,
    inset: 5pt,
    align: center + horizon,
    stroke: (x, y) => if y == 0 or y == 1 { 0.8pt } else { 0pt },
    table.header(..headers.map(h => text(weight: "bold", h))),
    ..range(rows).map(_ => range(columns).map(_ => [ ] )).flatten(),
    table.hline(stroke: 0.8pt),
  ),
)

// ---------- Identitas makalah ----------
#align(center)[
  #set par(
    first-line-indent: 0pt,
    justify: false,
  )

  #text(size: 15pt, weight: "bold")[
    Implementasi dan Eksperimen Komparatif Algoritma Particle Swarm Optimization (PSO) pada Optimasi Kontinu dan Seleksi Fitur Diskrit (BPSO)
  ]

  #v(0.6em)

  #text(size: 11pt)[
    Benedictus Ryu Gunawan#super[1]
  ]

  #v(0.3em)

  #text(size: 10pt)[
    #super[1] Rekayasa Kecerdasan Artifisial, Institut Teknologi Sepuluh Nopember
  ]
  #linebreak()

  #text(size: 10pt)[
    #super[#sym.ast]Corresponding author's email: #super[1]$"5054251001@student.its.ac.id"$
  ]
]

#v(0.6em)

#unnumbered-title[ABSTRAK]
Penelitian ini bertujuan untuk merancang, mengimplementasikan dari nol (*from scratch*), dan mengevaluasi performa algoritma *Particle Swarm Optimization* (PSO) dengan topologi global (*gbest*). Algoritma diuji pada dua fungsi benchmark optimasi kontinu berdimensi $D = 10$, yaitu fungsi Sphere (unimodal) dan Rastrigin (multimodal kompleks) pada domain $[-5.12, 5.12]^D$. Tiga varian strategi kontrol inersia dan konstriksi dievaluasi melalui 30 *independent runs*: bobot inersia tetap ($w = 0.7$), bobot inersia menurun linier ($w = 0.9 -> 0.4$), dan *constriction factor* Clerc-Kennedy ($chi = 0.7298$), seluruhnya dilengkapi *velocity clamping* dan *boundary handling*. Selain itu, sebagai tugas bonus, dikembangkan *Binary PSO* (BPSO) dengan fungsi transfer Sigmoid untuk seleksi fitur pada dataset riil UCI *Breast Cancer Wisconsin* (Diagnostic). Hasil eksperimen menunjukkan bahwa $w$ tetap ($w = 0.7$) paling unggul pada eksploitasi Sphere (rata-rata $1.25 times 10^(-11)$) sekaligus paling tangguh melompati *local traps* pada Rastrigin (rata-rata $8.39$). Pada seleksi fitur diskrit, BPSO berhasil mereduksi $66.7%$ dimensi fitur (dari 30 menjadi 10 fitur) sekaligus meningkatkan akurasi klasifikasi k-NN 5-fold CV dari $96.66%$ menjadi $97.72%$.

#v(0.35em)
#set par(first-line-indent: 0pt)
#strong[Kata kunci:] Particle Swarm Optimization (PSO), velocity clamping, boundary handling, constriction factor, fungsi benchmark, Binary PSO, seleksi fitur.

#set figure(supplement: "Gambar")
#show figure.caption: it => [
  *#it.supplement #it.counter.display(it.numbering).* #it.body
]

= Latar Belakang
Optimasi numerik terhadap fungsi objektif non-linier dan berdimensi tinggi merupakan salah satu tantangan fundamental dalam bidang komputasi cerdas dan rekayasa kecerdasan artifisial [1]. Berbagai metode optimasi klasik berbasis turunan (*gradient-based methods*) kerap menghadapi kendala serius ketika diterapkan pada lanskap pencarian yang multimodal, diskontinu, atau memiliki turunan yang sulit dievaluasi secara analitis. Oleh karena itu, pendekatan metaheuristik berbasis populasi (*population-based metaheuristics*) berkembang pesat sebagai alternatif yang tangguh (*robust*) dalam memecahkan permasalahan optimasi global.

Salah satu algoritma metaheuristik yang paling sukses dan luas diterapkan adalah *Particle Swarm Optimization* (PSO), yang diperkenalkan oleh James Kennedy dan Russell Eberhart pada tahun 1995 [1]. PSO terinspirasi dari perilaku sosial kawanan hewan, seperti kawanan burung (*bird flocking*) dan ikan (*fish schooling*), dalam mengoordinasikan pencarian sumber makanan. Dalam PSO, setiap partikel merepresentasikan kandidat solusi yang bergerak di dalam ruang pencarian berdimensi $D$ dengan kecepatan dinamis yang dipandu oleh *personal best* ($p b e s t$) dan *global best* ($g b e s t$).

Kinerja PSO sangat dipengaruhi oleh keseimbangan antara eksplorasi global (*exploration*) dan eksploitasi lokal (*exploitation*). Berbagai strategi kontrol inersia telah diajukan, antara lain bobot inersia konstan ($w = 0.7$), bobot inersia menurun linier ($w = 0.9 -> 0.4$) oleh Shi & Eberhart [2], dan faktor konstriksi (*constriction factor*) analitis ($chi = 0.7298$) oleh Clerc & Kennedy [3]. Selain itu, pencegahan ledakan kecepatan (*particle explosion*) melalui *velocity clamping* dan pengendalian batas (*boundary handling*) menjadi elemen krusial untuk menjamin stabilitas konvergensi. Di samping domain kontinu, PSO juga diadaptasi ke domain diskrit melalui *Binary PSO* (BPSO) untuk seleksi fitur (*feature selection*) [4]. Penelitian ini bertujuan mengimplementasikan PSO dari nol dengan NumPy, melakukan eksperimen 30 run pada fungsi Sphere dan Rastrigin ($D=10$), serta menerapkan BPSO pada dataset UCI *Breast Cancer Wisconsin* [5].

= Metodologi Penelitian
== Implementasi Algoritma PSO gbest dari Nol
Dalam topologi global (*gbest topology*), kecepatan dan posisi partikel $i$ diperbarui pada setiap iterasi $t$ menurut:
$ v_(i d)^((t+1)) = w(t) v_(i d)^((t)) + c_1 r_(1, i d) (p b e s t_(i d)^((t)) - x_(i d)^((t))) + c_2 r_(2, i d) (g b e s t_d^((t)) - x_(i d)^((t))) $
$ x_(i d)^((t+1)) = x_(i d)^((t)) + v_(i d)^((t+1)) $
di mana $r_1, r_2 tilde.op cal(U)(0, 1)$, $c_1$ koefisien kognitif, dan $c_2$ koefisien sosial. Tiga varian inersia yang dievaluasi:
1. *Bobot Inersia Tetap ($w = 0.7$):* Menjaga momentum konstan ($c_1 = c_2 = 1.49618$).
2. *Bobot Inersia Menurun Linier ($w = 0.9 -> 0.4$):* $w(t) = w_("max") - t / T_("max") (w_("max") - w_("min"))$ dengan $c_1 = c_2 = 2.0$ [2].
3. *Constriction Factor Clerc & Kennedy ($chi = 0.7298$):* $v_(i d)^((t+1)) = chi [v_(i d)^((t)) + phi_1 r_1 (p b e s t - x) + phi_2 r_2 (g b e s t - x)]$ dengan $phi = 4.1$ dan $chi = 0.7298$ [3].

== Velocity Clamping dan Penanganan Batas
- *Velocity Clamping:* Membatasi kecepatan pada $[-v_("max"), v_("max")]$ dengan $v_("max") = 0.2 times (x_("max") - x_("min")) = 2.048$ untuk mencegah divergensi.
- *Boundary Handling:* Meng-*clamp* posisi yang melanggar batas $[-5.12, 5.12]$ ke batas terdekat dan me-reset kecepatan menjadi nol ($v = 0$) saat membentur batas.

== Fungsi Uji Benchmark ($D = 10$)
1. *Fungsi Sphere (Unimodal):* $f_("Sphere")(x) = sum_(i=1)^D x_i^2$, optimum global $f(x^*) = 0$ pada $x^* = (0, dots, 0)$.
2. *Fungsi Rastrigin (Multimodal):* $f_("Rastrigin")(x) = 10 D + sum_(i=1)^D [x_i^2 - 10 cos(2 pi x_i)]$, memiliki ribuan *local minima*, optimum global $f(x^*) = 0$ pada $x^* = (0, dots, 0)$.

#figure(
  image("figures/lanskap_fungsi_uji.png", width: 60%),
  caption: [Kontur Lanskap Fungsi Uji 2D: Sphere vs Rastrigin],
) <fig-lanskap>

== Desain Eksperimen Komparatif
Setiap konfigurasi diuji sebanyak *30 independent runs* dengan *random seed* terkontrol ($1000 + 17i$), ukuran kawanan $N = 30$ partikel, dan iterasi maksimum $T_("max") = 200$.

== Bonus: Binary PSO (BPSO) untuk Seleksi Fitur
Solusi direpresentasikan sebagai vektor biner $x_i in {0, 1}^D$. Pemetaan kecepatan ke probabilitas bit menggunakan fungsi Sigmoid $S(v) = 1 / (1 + e^(-v))$. Posisi bit bernilai 1 jika $r < S(v)$, sebaliknya 0 [4]. Dataset yang digunakan adalah UCI *Breast Cancer Wisconsin* (569 sampel, 30 fitur kontinu terstandarisasi). Fungsi fitness multi-objektif meminimalkan error k-NN 5-fold CV dan rasio fitur:
$ "Fitness"(x) = alpha times (1 - "Akurasi"_(C V)) + (1 - alpha) times (|"Fitur Terpilih"| / "Total Fitur"), quad alpha = 0.95 $

= Hasil dan Pembahasan
== Analisis Statistik Benchmark
Hasil kuantitatif 30 *independent runs* dirangkum pada Tabel 1:

#figure(
  table(
    columns: (1fr, 1.8fr, 1.8fr, 1.2fr, 1.2fr, 1.2fr),
    inset: 3.5pt,
    align: center + horizon,
    stroke: 0.5pt,
    table.header(
      [*Fungsi*], [*Strategi Bobot*], [*Rata-rata $plus.minus$ Std*], [*Terbaik (Min)*], [*Median*], [*Terburuk (Max)*]
    ),
    [Sphere], [w Tetap (0.7)], [$1.247 times 10^(-11) plus.minus 1.645 times 10^(-11)$], [$3.601 times 10^(-13)$], [$6.425 times 10^(-12)$], [$6.386 times 10^(-11)$],
    [Sphere], [w Menurun (0.9 -> 0.4)], [$5.924 times 10^(-6) plus.minus 1.537 times 10^(-5)$], [$2.066 times 10^(-8)$], [$9.418 times 10^(-7)$], [$7.842 times 10^(-5)$],
    [Sphere], [Constriction ($chi=0.7298$)], [$6.831 times 10^(-10) plus.minus 8.745 times 10^(-10)$], [$9.983 times 10^(-11)$], [$4.120 times 10^(-10)$], [$4.512 times 10^(-9)$],
    [Rastrigin], [w Tetap (0.7)], [*$8.3920 plus.minus 3.2981$*], [*$1.9899$*], [*$7.9597$*], [$15.9193$],
    [Rastrigin], [w Menurun (0.9 -> 0.4)], [$10.0583 plus.minus 3.6106$], [$4.3153$], [$9.9496$], [$18.9042$],
    [Rastrigin], [Constriction ($chi=0.7298$)], [$9.9329 plus.minus 4.1376$], [$3.0755$], [$9.9496$], [$19.8992$],
  ),
  caption: [Ringkasan Statistik Nilai Fitness Terbaik (30 Run Independen, D = 10, T = 200)],
) <tab-benchmark>

#figure(
  image("figures/kurva_konvergensi_pso.png", width: 60%),
  caption: [Kurva Konvergensi Rata-rata 30 Run (Skala Logaritmik) dengan Pita Deviasi Standar ($plus.minus 1sigma$)],
) <fig-konvergensi>

#figure(
  image("figures/boxplot_fitness_pso.png", width: 60%),
  caption: [Boxplot Sebaran Nilai Fitness Terbaik Akhir dari 30 Run pada Sphere dan Rastrigin],
) <fig-boxplot>

== Analisis Konvergensi pada Sphere (Unimodal)
Pada fungsi Sphere, lanskap bersifat konveks tanpa hambatan lokal. Varian *$w$ Tetap ($w = 0.7$)* mencapai nilai rata-rata terbaik $1.247 times 10^(-11)$ dengan nilai terbaik $3.601 times 10^(-13)$. Inersia konstan $0.7$ menjaga osilasi partikel terus mengecil ke pusat lembah $(0, dots, 0)$. *Constriction Factor ($chi = 0.7298$)* juga sangat stabil ($6.831 times 10^(-10)$). Sebaliknya, *$w$ Menurun Linier* tertinggal di $5.924 times 10^(-6)$ karena penurunan $w$ hingga $0.4$ menyebabkan hilangnya momentum gerak (*loss of kinetic momentum*) sebelum eksploitasi sub-mikro tercapai.

== Analisis Ketahanan pada Rastrigin (Multimodal)
Pada fungsi Rastrigin, modulasi sinusoidal menghasilkan puluhan ribu *local optima*. Varian *$w$ Tetap ($w = 0.7$)* paling unggul dengan rata-rata terendah ($8.3920$) dan nilai terbaik $1.9899$. Kecepatan residual dari inersia $0.7$ memberi partikel energi kinetik untuk melompati bukit potensial lokal (*barrier hopping*). Sebaliknya, *Constriction Factor* ($9.9329$) dan *$w$ Menurun* ($10.0583$) mengalami deselerasi dini sehingga partikel membeku di cekungan lokal (*premature convergence*).

== Pengaruh Velocity Clamping dan Penanganan Batas
*Velocity clamping* ($v_("max") = 2.048$) mutlak mencegah fenomena *swarm explosion*, sementara *boundary handling* dengan *velocity zeroing* saat menyentuh batas domain berhasil mencegah partikel memantul liar di perbatasan.

== Evaluasi Seleksi Fitur Binary PSO pada Dataset UCI
Hasil seleksi fitur BPSO pada dataset UCI Breast Cancer dirangkum pada Tabel 2 dan Gambar 4:

#figure(
  table(
    columns: (2.2fr, 1.8fr, 1.8fr, 1.6fr),
    inset: 4pt,
    align: center + horizon,
    stroke: 0.5pt,
    table.header(
      [*Metode / Konfigurasi*], [*Jumlah Fitur*], [*Persentase Reduksi*], [*Akurasi 5-Fold CV*]
    ),
    [Baseline k-NN (Semua Fitur)], [30 / 30], [0.0%], [96.66% $plus.minus$ 1.40%],
    [Binary PSO (Fitur Terpilih)], [*10 / 30*], [*66.7%*], [*97.72%* (+$1.06%$)],
  ),
  caption: [Perbandingan Hasil Seleksi Fitur Baseline vs Binary PSO pada Dataset UCI],
) <tab-bpso>

#figure(
  image("figures/bpso_seleksi_fitur.png", width: 60%),
  caption: [Kurva Konvergensi BPSO dan Diagram Batang Reduksi Fitur vs Peningkatan Akurasi],
) <fig-bpso>

BPSO memangkas 66.7% dimensi fitur (dari 30 menjadi 10 fitur) dan meningkatkan akurasi dari 96.66% ke 97.72%. Fitur yang dipertahankan: `mean area`, `mean concavity`, `mean concave points`, `radius error`, `perimeter error`, `fractal dimension error`, `worst radius`, `worst texture`, `worst perimeter`, dan `worst area`. Fitur-fitur ini konsisten dengan indikator onkologi sel tumor ganas, di mana eliminasi 20 fitur lainnya menyingkirkan multikolinearitas dan *noise*.

= Kesimpulan
1. *Strategi Inersia & Konstriksi:* Varian $w$ tetap ($0.7$) paling seimbang dan unggul pada Sphere ($1.25 times 10^(-11)$) maupun Rastrigin ($8.392$) karena menjaga energi kinetik partikel agar terhindar dari konvergensi prematur.
2. *Velocity Clamping & Boundary Handling:* Terbukti esensial dalam menjaga stabilitas lintasan kawanan partikel dan mencegah fenomena divergensi.
3. *Keberhasilan Binary PSO (Bonus):* BPSO mereduksi 66.7% fitur dataset UCI Breast Cancer (dari 30 ke 10 fitur) dengan peningkatan akurasi klasifikasi menjadi 97.72%.

#heading(numbering: none, level: 1)[Daftar Pustaka]
#set par(first-line-indent: 0pt, hanging-indent: 0.7cm)

[1] J. Kennedy and R. Eberhart, "Particle swarm optimization," in _Proceedings of ICNN'95 - International Conference on Neural Networks_, Perth, WA, Australia, 1995, vol. 4, pp. 1942-1948.
#linebreak()

[2] Y. Shi and R. Eberhart, "A modified particle swarm optimizer," in _1998 IEEE International Conference on Evolutionary Computation Proceedings_, Anchorage, AK, USA, 1998, pp. 69-73.
#linebreak()

[3] M. Clerc and J. Kennedy, "The particle swarm - explosion, stability, and convergence in a multidimensional complex space," _IEEE Transactions on Evolutionary Computation_, vol. 6, no. 1, pp. 58-73, Feb. 2002.
#linebreak()

[4] J. Kennedy and R. C. Eberhart, "A discrete binary version of the particle swarm algorithm," in _1997 IEEE International Conference on Systems, Man, and Cybernetics_, Orlando, FL, USA, 1997, vol. 5, pp. 4104-4108.
#linebreak()

[5] W. N. Street, W. H. Wolberg, and O. L. Mangasarian, "Nuclear feature extraction for breast tumor diagnosis," in _IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology_, San Jose, CA, USA, 1993, vol. 1905, pp. 861-870.
