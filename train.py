import pygame
import config.display as display
import utils.UI as UI

def train_ddqn(agent, cartpole, render=True):
    if render:
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((display.WIDTH, display.HEIGHT))

    iteration = 0    
    done = False
    # not done with training
    while not done:
        end_episode = False
        # the current episode did not end
        while not end_episode:
            if render:
                clock.tick(display.FPS)
                current_time = pygame.time.get_ticks()
                elapsed_time = (current_time - clock.get_time()) / 1000 # convert ms into s
                for event in pygame.event.get(): # event handling
                    if event.type == pygame.QUIT:
                        done = True
                        end_episode = True
                        break
            
            # get current state and current actions
            current_state = cartpole.get_state()
            current_action = agent.get_action(current_state)
            
            # reward from the action
            reward, game_over = cartpole.move(action=current_action)

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
                UI.render(screen, cartpole, elapsed_time, iteration)
                pygame.display.flip()

        # plot graphs after episode ends
        cartpole.plot_score()
        agent.plot_model()

        # save the model if new average is higher
        if cartpole.old_mean < cartpole.new_mean:
            agent.save_model(iteration, cartpole.new_mean)

    if render:
        pygame.quit()