import random ,sys
import numpy as np


#Network
INPUT_SIZE = 16
HIDDEN_SIZE = 10
OUTPUT_SIZE = 1

#Genetic algorithm
GENERATIONS = 80
POPULATION_SIZE = 120
REPLACEMENT_SIZE = int(POPULATION_SIZE * 0.4)

#The Network - 3 layers : input 16, hidden 10, output 1
class Network:
    def __init__(self):
        self.HiddenLayerW = np.random.rand(INPUT_SIZE, HIDDEN_SIZE) * 0.5
        self.OutputLayerW = np.random.rand(HIDDEN_SIZE, OUTPUT_SIZE) * 0.5

    #The move on the Network
    def Forward(self, inputs):
        HiddenLayer = np.dot(inputs, self.HiddenLayerW)
        ActivationHiddenLayer = self.Sigmoid(HiddenLayer)
        OutputLayer = np.dot(ActivationHiddenLayer, self.OutputLayerW)
        output = self.Sigmoid(OutputLayer)
        return 1 if output > 0.5 else 0

    #Activation function
    def Sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

#Genetic algorithm
class GeneticAlgorithm:
    def __init__(self, TestData, TrainData):
        self.TrainData = TrainData
        self.TestData = TestData
        self.Mutation = 0.2
        self.Network = Network()

    # function calculates the fitness score for a given set of weights by evaluating the neural network's predictions
    # on the training data and comparing them to the actual target values
    def CalculateFitness(self, weights):
        self.SplitWeigth(weights)
        PredictionsScore = sum(self.Network.Forward(np.array(data[:-1], dtype=float)) == np.array(data[-1], dtype=float)
                               for data in self.TrainData)
        return PredictionsScore / len(self.TrainData)

    #The Crossover function performs a crossover operation between two parents by randomly selecting a crossover point,
    # and returns the resulting offspring.
    def ChossePerant(self,population, FitnessScores):
        TournamentCandidates = random.sample(range(len(population)), 5)
        TournamentScores = [FitnessScores[i] for i in TournamentCandidates]
        return population[TournamentCandidates[TournamentScores.index(max(TournamentScores))]]


    #This function gives weigte for the populiten in the range [-1,1]
    def CreatePopulation(self):
        population = [np.random.uniform(low=-1, high=1, size=self.Network.HiddenLayerW.size +
                                                             self.Network.OutputLayerW.size) for _ in range(POPULATION_SIZE)]
        return population

    #This function split the weigth for two layers- for hiddenen and output
    def SplitWeigth(self, weights):
        HiddenLayerW = np.reshape(weights[:self.Network.HiddenLayerW.size], self.Network.HiddenLayerW.shape)
        OutputLayerW = np.reshape(weights[self.Network.HiddenLayerW.size:], self.Network.OutputLayerW.shape)
        self.Network.HiddenLayerW = HiddenLayerW
        self.Network.OutputLayerW = OutputLayerW

    #The Crossover function performs a crossover operation between two parents by randomly selecting a crossover point,
    # and returns the resulting offspring.
    def Crossover(self, parent1, parent2):
        CrossoverPoint = random.randint(1, len(parent1) - 1)
        return  np.concatenate((parent1[:CrossoverPoint], parent2[CrossoverPoint:])), np.concatenate((parent2[:CrossoverPoint], parent1[CrossoverPoint:]))


    # Create a mutae-  function applies random mutations to the given weights array,
    # ensuring that the mutated values stay within a specific range of -1 to 1.
    def Mutate(self, weights):
        MutateWeights = np.copy(weights)
        mask = np.random.random(size=weights.shape) < self.Mutation
        mutations = np.random.normal(loc=-0.1, scale=0.1, size=weights.shape)
        MutateWeights[mask] += mutations[mask]
        MutateWeights = np.clip(MutateWeights, -1, 1)
        return MutateWeights

    #Replace weakest weights with the mutated
    def ReplacePopulation(self, population, AllChaild, fitness_scores):
        worst_indices = np.argsort(fitness_scores)[:3*REPLACEMENT_SIZE] # Get worst by fitness scores
        fittest_indices = np.argsort([self.CalculateFitness(child) for child in AllChaild])[::-1][:REPLACEMENT_SIZE]
        # Replace individuals in the population with fittest offspring
        for i, fittest_index in enumerate(fittest_indices):
            population[worst_indices[i]] = AllChaild[fittest_index]
        return population

    #The main function
    def run(self):
        FittestW = []
        BestFitness ,FitnessCount = 0 ,0
        population = self.CreatePopulation()

        #loop over the generation numbers
        for generation in range(GENERATIONS):
            Scores,AllChild = [] ,[]

            #get the fitness for each row of wegite in the population
            for weights in population:
                fitness = self.CalculateFitness(weights)
                Scores.append(fitness)

            #get new wegith
            for _ in range(POPULATION_SIZE//2):
                parent1, parent2 = self.ChossePerant(population, Scores) ,self.ChossePerant(population, Scores)
                CrossoverPoint = random.randint(1, len(parent1) - 1)
                AllChild.append(self.Mutate(np.concatenate((parent1[:CrossoverPoint], parent2[CrossoverPoint:]))))
                AllChild.append(self.Mutate(np.concatenate((parent2[:CrossoverPoint], parent1[CrossoverPoint:]))))


            population = self.ReplacePopulation(population, AllChild, Scores)#Replace weakest weights with the mutated

            FittestWInd, FittestW = max(enumerate(population), key=lambda x: Scores[x[0]])# Update weights
            self.SplitWeigth(FittestW)
            BestFitnessScores = Scores[FittestWInd]
            print("Generation -> ",generation, "|  Best Fitness -> ",BestFitnessScores)

            FitnessCount = FitnessCount + 1 if BestFitnessScores == BestFitness else 0
            if FitnessCount ==0:
                self.Mutation = 0.2
                BestFitness = BestFitnessScores
            if FitnessCount > 12:
                self.Mutation = 0.9
        return FittestW

# Save weights to a file
def SaveToFile(path, weights):
    HiddenLayerW = weights[:HIDDEN_SIZE * INPUT_SIZE]
    OutputLayerW = weights[HIDDEN_SIZE * INPUT_SIZE:]
    data = [str(INPUT_SIZE) + ',' + str(HIDDEN_SIZE) + ',' + str(OUTPUT_SIZE),
            '[' + ', '.join(map(str, HiddenLayerW.flatten())) + ']',
            '[' + ', '.join(map(str, OutputLayerW.flatten())) + ']']
    with open(path, 'w') as file:
        file.write('\n'.join(data))

#Insert the train& test data in a arry structior
def ReadData(file_path):
    Data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            line = line.replace(' ', '')
            Data.append(np.array(list(map(int, filter(str.isdigit, line))), dtype=int))
    return Data


def SplitTextFile(input_file, num_lines, namefile1, namefile2):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Split the lines into two lists
    lines1 = lines[:num_lines]
    lines2 = lines[num_lines:]

    # Write lines1 to file1
    with open(namefile1, 'w') as file1:
        file1.writelines(lines1)

    # Write lines2 to file2
    with open(namefile2, 'w') as file2:
        file2.writelines(lines2)

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) > 0:
        TrainData = args[0]
        if len(args) > 1:
            TestData = args[1]
    else:
        SplitTextFile("nn1.txt",14000,"nn1Train.txt","nn1Test.txt")
        TrainData =ReadData("nn1train.txt")
        TestData = ReadData("nn1Test.txt")
    GeneticAlgorithm = GeneticAlgorithm(TestData, TrainData)
    Weigths = GeneticAlgorithm.run()
    #save in file
    SaveToFile("wnet1.txt", Weigths)