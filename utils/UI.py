import pygame
import config.display as display

pygame.init()
font = pygame.font.Font("Arial.ttf", 20)

def render(screen, cartpole, iteration=None):
    screen.fill(display.BLACK)
    cartpole.draw(screen)

    # Render texts
    draw_text(screen, f"Score: {cartpole.current_score} s", (0, 0))
    draw_text(screen, f"deaths: {cartpole.num_episodes}", (0, 25))
    draw_text(screen, f"Best score: {cartpole.best_score} s", (0, 50))

    if iteration != None:
        draw_text(screen, f"Iterations: {iteration}", (0, 75))

def draw_text(screen, text, position, antialias=True, color = display.WHITE):
    text_surface = font.render(text, antialias, color)
    screen.blit(text_surface, position)