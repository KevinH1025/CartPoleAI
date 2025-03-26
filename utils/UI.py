import pygame
import logic.settings as settings

pygame.init()
font = pygame.font.Font("Arial.ttf", 20)

def render(screen, cartpole, elapsed_time, iteration=None):
    screen.fill(settings.BLACK)
    cartpole.draw(screen)

    # Render texts
    draw_text(screen, f"Score: {cartpole.current_score}", (0, 0))
    draw_text(screen, f"deaths: {cartpole.num_episodes}", (0, 25))
    draw_text(screen, f"Best score: {cartpole.best_score}", (0, 50))

    draw_text(screen, f"Time: {elapsed_time}s", (0, 75))
    if iteration != None:
        draw_text(screen, f"Iterations: {iteration}", (0, 100))

def draw_text(screen, text, position, antialias=True, color = settings.WHITE):
    text_surface = font.render(text, antialias, color)
    screen.blit(text_surface, position)