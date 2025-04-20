import pygame
import config.display as display
import utils.UI as UI
from models.DDQN import DDQN_Agent
from cartPole import CartPole
from torch.utils.tensorboard import SummaryWriter

def train_agent(algorithm, render=True, plot=True):
    if plot:
        writer = SummaryWriter()

    cartpole = CartPole(plot, 'train')
    
    print("Training model...")
    if algorithm == "ddqn":
        agent = DDQN_Agent(len(cartpole.get_state()), plot)
        train_ddqn(agent, cartpole, writer, render)
    elif algorithm == "ppo":
        print("PPO training is not implemented yet")
        return
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    if writer:
        writer.close()
    
    print("Training stopped.")

def train_ddqn(agent:DDQN_Agent, cartpole:CartPole, writer, render):
    if render:
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((display.WIDTH, display.HEIGHT))

    iteration = 0    
    done = False
    # not done with training
    while not done:
        start_time = pygame.time.get_ticks()
        end_episode = False

        # the current episode did not end
        while not end_episode:
            if render:
                clock.tick(display.FPS)
                current_time = pygame.time.get_ticks()
                elapsed_time = int((current_time - start_time) / 1000) # convert ms to s
                for event in pygame.event.get(): # event handling
                    if event.type == pygame.QUIT:
                        done = True

            # close the program if the user clicks on the close
            if done:
                break
            
            # get current state and current actions
            current_state = cartpole.get_state()
            current_action = agent.get_action(current_state)
            
            # reward from the action
            reward, game_over = cartpole.move(elapsed_time, action=current_action)

            # episode ended?
            end_episode = game_over

            # get next state
            next_state = cartpole.get_state()

            # save it to the buffer
            agent.store_experience(current_state, current_action, reward, next_state, game_over)
            
            # train the agent
            if agent.train_process() != False and render == True:
                iteration += 1 # one train iteration

            # render the cartpole
            if render:
                UI.render(screen, cartpole, iteration)
                pygame.display.flip()

        # plot graphs after episode ends
        cartpole.plot_score(writer)
        agent.plot_model(writer)

        # save the model if new average is higher
        if cartpole.old_mean < cartpole.new_mean:
            agent.save_model(iteration, cartpole.new_mean)

    if render:
        pygame.quit()