*** (a)
gen cratio_1 = cratio[_n-1]
gen rrate_1 = rrate[_n-1]
gen e_1 = e[_n-1]

*** GMM 
gmm({beta=1}* (cratio^(-1*{gamma=1}))*rrate-1), inst(cratio_1 rrate_1 e_1)

***(b)
gen time=[_n]
gmm({beta=1}* (cratio^(-1*{gamma=1}))*rrate-1), inst(cratio_1 rrate_1 e_1) wmatrix(hac ba 5)