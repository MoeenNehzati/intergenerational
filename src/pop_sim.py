import numpy as np 
from tqdm import tqdm
from numpy import sqrt 
np.random.seed(11523)
pop_size = 1000000
length = 10
number_of_the_genes = 100
vg0 = np.eye(number_of_the_genes)/number_of_the_genes
# a = np.random.normal(size = (1, number_of_the_genes))
a = np.array([[-0.8861699891156921, 1.1228465347731809, -0.45386290670174556, -2.157947755560483, 0.5686769396155648, 1.1801789648434735, -0.9362778994723663, -0.4493046477448862, 1.4257687863078319, 0.32639297034238995, 0.9991524658086587, 1.2463675145907565, 0.44492190272087623, -1.281204760796021, 0.7136910764683282, -0.8362945067705422, -0.04801139124905767, -1.9571534131527033, 1.4884079643132953, -0.785809489143734, -1.033410159867713, 0.2128786715199647, 0.10914583768049659, -0.5505775822721471, -0.383588267368759, 0.4637729325579148, -1.552001217276316, 0.5536337416108499, -0.9318447743149747, 0.12295979431539457, 0.8370310216930527, 0.1406700064225533, 0.2554688504137627, 1.2670688498690281, -1.8437228756194348, -0.9493526433679758, -0.9221762243338267, -0.46358189249183096, 0.3202651931400246, 0.8606172137583035, 0.6141235208141975, 0.34175703412743147, -0.6075480214183002, -0.2078796103212297, 0.07839777676271567, 0.3531187602719548, -0.7567066328314149, 0.3959582242409504, 0.26038102006213965, -0.9550484097794422, -0.8222800118193825, -1.1586903514789497, -0.36647568434083627, 1.0915206073622499, -1.436100573419577, 0.17062153485941076, -0.10180618086182189, -0.8661915644264963, -0.9178813852683909, -0.7727211335892226, 1.7893741797724085, 0.3424490926932624, -0.7267166123804576, 0.1436161603491659, 0.0827167883805811, -0.4402449725217179, 0.8766435180474552, 0.5535933545125449, -0.5467644764334935, -0.41554548026171517, 0.43824724965304296, 0.9426658084591903, -0.08264708151340687, 0.06320609061404367, 0.7202890824263617, -0.6079253160016463, -0.4440595558836188, 0.6443343960386642, -1.4364889240615886, 1.3186816400075618, 1.467518788581399, -1.6079016276590394, 0.15588648215202536, 0.24235296345757973, 1.369260272698099, -0.6274235132876304, 1.0619633276081113, -0.06838521244813235, -1.7501508878317253, 1.4241052068245075, -0.027009633046335017, 0.20253586890205977, 1.2432874575655186, 0.4415613928533352, 1.0712711903837646, -0.9505355988379756, 0.5793800287648175, -0.6872798018855655, -0.16265047495584806, 0.8627903065642525]])
p = 0.5
mu = 1
ve = 1
f = 0
for i in np.roots([2, -2, vg0[0, 0]]):
    if 0 < i <= 0.5:
        f = i
        break

if f == 0:
    raise Exception()


males = [None]*length
females = [None]*length
male_phenotypes = [None]*length
female_phenotypes = [None]*length
vy = [None]*length
corrs = [None]*length

males[0] = np.random.binomial(2, f, (number_of_the_genes, pop_size//2))
females[0] = np.random.binomial(2, f, (number_of_the_genes, pop_size//2))
male_phenotypes[0] = a@males[0] + np.random.normal(0, sqrt(ve), (1, pop_size//2))
female_phenotypes[0] = a@females[0] + np.random.normal(0, sqrt(ve), (1, pop_size//2))
vy[0] = np.var(np.hstack((male_phenotypes[0][0], female_phenotypes[0][0])))

for i in tqdm(range(1, length)):
    if i >= 2:
        males[i-2] = None
        males[i-2] = None
    father_phenotype = male_phenotypes[i-1]
    mother_phenotype = female_phenotypes[i-1]
    father = males[i-1]
    mother = females[i-1]
    noisy_father_phenotype = father_phenotype + np.random.normal(0, sqrt((1/p - 1)*vy[i-1]), size=father_phenotype.shape)
    noisy_mother_phenotype = mother_phenotype + np.random.normal(0, sqrt((1/p - 1)*vy[i-1]), size=mother_phenotype.shape)
    father = father[:, np.argsort(noisy_father_phenotype)[0]]
    mother = mother[:, np.argsort(noisy_mother_phenotype)[0]]
    father_phenotype_sorted = father_phenotype[0][np.argsort(noisy_father_phenotype)]
    mother_phenotype_sorted = mother_phenotype[0][np.argsort(noisy_mother_phenotype)]
    corrs[i-1] = np.corrcoef(father_phenotype_sorted, mother_phenotype_sorted)

    son_fathers = father[:, [i%2==0 for i in range(father.shape[1])]]
    son_mothers = mother[:, [i%2==0 for i in range(mother.shape[1])]]
    daughter_fathers = father[:, [i%2!=0 for i in range(father.shape[1])]]
    daughter_mothers = mother[:, [i%2!=0 for i in range(mother.shape[1])]]
    son1 = (son_fathers==2) + (son_fathers==1)*np.random.binomial(1, 0.5, son_fathers.shape) + (son_mothers==2) + (son_mothers==1)*np.random.binomial(1, 0.5, son_mothers.shape)
    son2 = (son_fathers==2) + (son_fathers==1)*np.random.binomial(1, 0.5, son_fathers.shape) + (son_mothers==2) + (son_mothers==1)*np.random.binomial(1, 0.5, son_mothers.shape)
    daughter1 = (daughter_fathers==2) + (daughter_fathers==1)*np.random.binomial(1, 0.5, daughter_fathers.shape) + (daughter_mothers==2) + (daughter_mothers==1)*np.random.binomial(1, 0.5, daughter_mothers.shape)
    daughter2 = (daughter_fathers==2) + (daughter_fathers==1)*np.random.binomial(1, 0.5, daughter_fathers.shape) + (daughter_mothers==2) + (daughter_mothers==1)*np.random.binomial(1, 0.5, daughter_mothers.shape)

    males[i] = np.hstack((son1, son2))
    females[i] = np.hstack((daughter1, daughter2))

    male_phenotypes[i] = a@males[i] + mu*a@(np.hstack((son_fathers, son_fathers)) + np.hstack((son_mothers, son_mothers))) + np.random.normal(0, sqrt(ve), (1, pop_size//2))
    female_phenotypes[i] = a@females[i] + mu*a@(np.hstack((daughter_fathers, daughter_fathers)) + np.hstack((daughter_mothers, daughter_mothers))) + np.random.normal(0, sqrt(ve), (1, pop_size//2))
    vy[i] = np.var(np.hstack((male_phenotypes[i][0], female_phenotypes[i][0])))























# males = [np.random.binomial(2, f, (number_of_the_genes, pop_size//2))]
# females = [np.random.binomial(2, f, (number_of_the_genes, pop_size//2))]
# male_phenotypes = [a @ males[0] + np.random.normal(0, sqrt(ve), size=(1, pop_size//2))]
# female_phenotypes = [a @ females[0] + np.random.normal(0, sqrt(ve), size=(1, pop_size//2))]
# vy = [np.var(np.hstack([male_phenotypes, female_phenotypes]))]
# corrs = []
# father_ranks = []
# mother_ranks = []
# vg = [2*f*(1-f)]
# for i in tqdm(range(length)):
#     noisy_fathers_y = male_phenotypes[i] + np.random.normal(0, sqrt((1/p-1)*np.var(male_phenotypes[i])), size=(1, pop_size//2))
#     noisy_mothers_y = female_phenotypes[i] + np.random.normal(0, sqrt((1/p-1)*np.var(female_phenotypes[i])), size=(1, pop_size//2))
#     father_ranks = [index for index, y in sorted(enumerate(noisy_fathers_y[0].tolist()), key = lambda x:x[1])]
#     mother_ranks = [index for index, y in sorted(enumerate(noisy_mothers_y[0].tolist()), key = lambda x:x[1])]
#     corrs.append(np.corrcoef(male_phenotypes[i][0, father_ranks], female_phenotypes[i][0, mother_ranks]))

#     fathers = males[i][:,father_ranks]
#     mothers = females[i][:,mother_ranks]

#     f_families = [i%2 == 0 for i in range(pop_size//2)]
#     m_families = [i%2 == 1 for i in range(pop_size//2)]
#     m_father = fathers[:, m_families]
#     m_mother = mothers[:, m_families]
#     f_father = fathers[:, f_families]
#     f_mother = mothers[:, f_families]

#     new_males1 = (m_father==2) + np.random.binomial(1, 0.5, m_father.shape)*(m_father==1) + (m_mother==2) + np.random.binomial(1, 0.5, m_mother.shape)*(m_mother==1)
#     print("COOV", np.corrcoef(new_males1.reshape((1, -1)), fathers[:, m_families].reshape((1, -1))))
#     new_males2 = (m_father==2) + np.random.binomial(1, 0.5, m_father.shape)*(m_father==1) + (m_mother==2) + np.random.binomial(1, 0.5, m_mother.shape)*(m_mother==1)
#     new_males = np.hstack((new_males1, new_males2))
#     new_females1 = (f_father==2) + np.random.binomial(1, 0.5, f_father.shape)*(f_father==1) + (f_mother==2) + np.random.binomial(1, 0.5, f_mother.shape)*(f_mother==1)
#     new_females2 = (f_father==2) + np.random.binomial(1, 0.5, f_father.shape)*(f_father==1) + (f_mother==2) + np.random.binomial(1, 0.5, f_mother.shape)*(f_mother==1)
#     new_females = np.hstack((new_females1, new_females2))

#     # new_females = (fathers==2) + np.random.binomial(1, 0.5, (number_of_the_genes, pop_size//2))*(fathers==1) + (mothers==2) + np.random.binomial(1, 0.5, (number_of_the_genes, pop_size//2))*(mothers==1)
#     new_male_phenotypes = a @ new_males + mu*a@(fathers+mothers) + np.random.normal(0, sqrt(ve), size=(1, pop_size//2))
#     new_female_phenotypes = a @ new_females + mu*a@(fathers+mothers) + np.random.normal(0, sqrt(ve), size=(1, pop_size//2))
#     male_phenotypes.append(new_male_phenotypes)
#     female_phenotypes.append(new_female_phenotypes)
#     males.append(new_males)
#     females.append(new_males)
#     vy.append(np.var(np.hstack([new_male_phenotypes, new_female_phenotypes])))
#     vg.append(np.var(np.hstack([new_males, new_females]).reshape((1,-1))))
    
# print(vy)