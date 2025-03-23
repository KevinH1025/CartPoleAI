import pygame
import settings
import UI

def train_ddqn(agent, cartpole, render=True):
    done = False

    if render:
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((settings.WIDTH, settings.HEIGHT))
        iteration = 0

    while not done:
        if render:
            clock.tick(settings.FPS)
            current_time = pygame.time.get_ticks()
            elapsed_time = (current_time - clock.get_time()) / 1000 # convert ms into s
            for event in pygame.event.get(): # event handling
                if event.type == pygame.QUIT:
                    done = True
                    break
        
        # get current state and current actions
        current_state = cartpole.get_state()
        current_action = agent.get_action(current_state)
        
        # reward from the action
        reward, game_over = cartpole.move(action=current_action)

        # get next state
        next_state = cartpole.get_state()

        # save it to the buffer
        agent.store_experience(current_state, current_action, reward, next_state, game_over)
        
        # train the agent
        if agent.train_process() != False and render == True:
            iteration += 1

        if render:
            UI.render(screen, cartpole, elapsed_time, iteration)
            pygame.display.flip()
    
    if render:
        pygame.quit()