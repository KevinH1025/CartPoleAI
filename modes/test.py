from cartPole import CartPole
from models.DDQN import DDQN_Agent
import pygame
import config.display as display
import utils.UI as UI

def test_agent(algorithm):
    cartpole = CartPole(plot=True)
    # load the corresponding model
    print("Loading model...")
    if algorithm == "ddqn":
        agent = DDQN_Agent(len(cartpole.get_state()), plot=False)
        agent.load_model("trained_models/DDQN_model/20250330_012727_Reward_Shaping")
    else:
        pass

    test_model(cartpole, agent)

# test the model 
def test_model(cartpole, agent):
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((display.WIDTH, display.HEIGHT))
    done = False

    # not done with testing
    while not done:
        start_time = pygame.time.get_ticks()
        end_episode = False

        # the current episode did not end
        while not end_episode:
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
            current_action = agent.get_action(current_state, test=True)
            
            # update cart's movement
            _, game_over = cartpole.move(elapsed_time, action=current_action)

            # episode ended?
            end_episode = game_over

            # render the cartpole
            UI.render(screen, cartpole)
            pygame.display.flip()

        # plot score after episode ends
        cartpole.plot_score()

    pygame.quit()
