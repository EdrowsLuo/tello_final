import pygame


def main():
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    screen.fill((30, 30, 30))
    s = pygame.Surface(screen.get_size(), pygame.SRCALPHA, 32)
    pygame.draw.line(s, (0, 0, 0), (250, 250), (250 + 200, 250))

    screen.blit(s, (0, 0))

    pygame.draw.circle(screen, pygame.Color("GREEN"), (50, 100), 10)
    pygame.draw.circle(screen, pygame.Color("BLACK"), (50, 100), 10, 1)
    pygame.draw.polygon(screen, (255, 255, 255), [(100, 100), (200, 100), (200, 400)], 0)

    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            print(event)
            if event.type is pygame.QUIT:
                exit()


if __name__ == '__main__':
    main()
