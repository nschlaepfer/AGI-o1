import pygame
import random

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Catch the Square")

# Define colors
WHITE = (255, 255, 255)
PLAYER_COLOR = (50, 168, 82)
TARGET_COLOR = (168, 50, 50)

# Set up the player
player_size = 50
player_pos = [WIDTH // 2, HEIGHT - 2 * player_size]
player_speed = 5

# Set up the target
target_size = 50
target_pos = [random.randint(0, WIDTH - target_size), 0]
target_speed = 3

# Set up the clock
clock = pygame.time.Clock()

# Score
score = 0
font = pygame.font.SysFont("monospace", 35)

# Game loop
running = True
while running:
    clock.tick(60)  # Frames per second

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move the player
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and player_pos[0] > 0:
        player_pos[0] -= player_speed
    if keys[pygame.K_RIGHT] and player_pos[0] < WIDTH - player_size:
        player_pos[0] += player_speed

    # Move the target down
    target_pos[1] += target_speed

    # Reset the target when it goes off-screen
    if target_pos[1] > HEIGHT:
        target_pos = [random.randint(0, WIDTH - target_size), 0]

    # Check for collision
    player_rect = pygame.Rect(player_pos[0], player_pos[1], player_size, player_size)
    target_rect = pygame.Rect(target_pos[0], target_pos[1], target_size, target_size)
    if player_rect.colliderect(target_rect):
        score += 1
        target_pos = [random.randint(0, WIDTH - target_size), 0]
        # Increase difficulty
        target_speed += 0.5

    # Drawing
    window.fill(WHITE)

    # Draw player and target
    pygame.draw.rect(window, PLAYER_COLOR, player_rect)
    pygame.draw.rect(window, TARGET_COLOR, target_rect)

    # Display score
    score_text = font.render(f"Score: {score}", True, (0, 0, 0))
    window.blit(score_text, (10, 10))

    # Update the display
    pygame.display.update()

# Cleanup
pygame.quit()