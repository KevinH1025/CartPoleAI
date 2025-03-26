import matplotlib.pyplot as plt

def plot_score(score, mean_score):
    plt.figure("Score plot")  
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Number of deaths")
    plt.ylabel("Score")
    plt.plot(range(1, len(score) + 1), list(score), label="Score per death", color="blue", marker='o')
    plt.plot(range(1, len(mean_score) + 1), list(mean_score), label="Mean score", color="orange", marker='x')
    plt.legend()
    plt.ylim(ymin=0)
    plt.text(len(score)-1, score[-1], str(score[-1])) # write the value of the last element
    plt.text(len(mean_score)-1, mean_score[-1], str(mean_score[-1])) # write the value of the last element
    plt.pause(0.001)

# plot the loss
def plot_loss(loss, average_loss):
    plt.figure("Loss plot")  
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.plot(range(1, len(loss) + 1), list(loss), label="Current loss", color="blue", marker='o')
    plt.plot(range(1, len(average_loss) + 1), list(average_loss), label="Average loss", color="orange", marker='x')
    plt.legend()
    plt.ylim(ymin=0)
    plt.text(len(loss)-1, loss[-1], str(loss[-1])) # write the value of the last element
    plt.text(len(average_loss)-1, average_loss[-1], str(average_loss[-1])) # write the value of the last element
    plt.pause(0.001)

def plot_qvalue(avg_qvalues):
    plt.figure("Q-values plot")  
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Number of iterations")
    plt.ylabel("Q-values")
    plt.plot(range(1, len(avg_qvalues) + 1), list(avg_qvalues), label="Average Q-values", color="blue", marker='o')
    plt.legend()
    plt.ylim(ymin=0)
    plt.text(len(avg_qvalues)-1, avg_qvalues[-1], str(avg_qvalues[-1])) # write the value of the last element
    plt.pause(0.001)

def plot_epsilon(epsilon):
    plt.figure("Epsilon plot")  
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Number of iterations")
    plt.ylabel("Epsilon")
    plt.plot(range(1, len(epsilon) + 1), list(epsilon), label="Epsilon values", color="blue", marker='o')
    plt.legend()
    plt.ylim(ymin=0)
    plt.text(len(epsilon)-1, epsilon[-1], str(epsilon[-1])) # write the value of the last element
    plt.pause(0.001)