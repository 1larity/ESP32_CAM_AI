"""
Multi-Feed AI Viewer with Overlay Control

Author: Stellaris
Date: 2025-08-05
"""

import pygame
import tkinter as tk
from tkinter import simpledialog
import json
from .camera_feed import CameraFeed
from .feed_window import FeedWindow
from .constants import WINDOW_SIZE, LAYOUT_FILE

def prompt_for_url():
    root = tk.Tk()
    root.withdraw()
    return simpledialog.askstring("New Feed", "Enter MJPEG stream URL:")

def save_layout(windows):
    layout = [{"name": w.name, "x": w.rect.x, "y": w.rect.y, "w": w.rect.width, "h": w.rect.height, "url": w.feed.url} for w in windows]
    with open(LAYOUT_FILE, 'w') as f:
        json.dump(layout, f)

def load_layout():
    try:
        with open(LAYOUT_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
    pygame.display.set_caption("Multi-Feed AI Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 16)

    feeds, windows = {}, []
    for item in load_layout():
        feeds[item['name']] = feeds.get(item['name']) or CameraFeed(item['name'], item['url'])
        windows.append(FeedWindow(item['name'], feeds[item['name']], item['x'], item['y'], item.get('w', 320), item.get('h', 240)))

    menu_rects = {
        'New': pygame.Rect(10, 5, 60, 25),
        'Save': pygame.Rect(80, 5, 60, 25),
        'Load': pygame.Rect(150, 5, 60, 25),
        'Remove': pygame.Rect(220, 5, 90, 25),
        'Quit': pygame.Rect(320, 5, 60, 25)
    }

    running = True
    while running:
        screen.fill((30, 30, 30))

        for event in pygame.event.get():
            for w in windows:
                w.handle_event(event)
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                for label, rect in menu_rects.items():
                    if rect.collidepoint(event.pos):
                        if label == 'New':
                            url = prompt_for_url()
                            if url:
                                name_base = "Cam"
                                index = 1
                                while f"{name_base}{index}" in feeds:
                                    index += 1
                                name = f"{name_base}{index}"
                                feed = CameraFeed(name, url)
                                feeds[name] = feed
                                start_x = 30 + (index * 40) % (WINDOW_SIZE[0] - 350)
                                start_y = 50 + (index * 40) % (WINDOW_SIZE[1] - 250)
                                windows.append(FeedWindow(name, feed, start_x, start_y))
                        elif label == 'Save':
                            save_layout(windows)
                        elif label == 'Load':
                            windows.clear()
                            for item in load_layout():
                                feeds[item['name']] = feeds.get(item['name']) or CameraFeed(item['name'], item['url'])
                                windows.append(FeedWindow(item['name'], feeds[item['name']], item['x'], item['y'], item.get('w', 320), item.get('h', 240)))
                        elif label == 'Remove':
                            windows = [w for w in windows if not w.selected]
                        elif label == 'Quit':
                            running = False

        for w in windows:
            w.draw(screen, font)

        for label, rect in menu_rects.items():
            pygame.draw.rect(screen, (70, 70, 70), rect)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)
            screen.blit(font.render(label, True, (255, 255, 255)), (rect.x + 5, rect.y + 5))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == '__main__':
    main()
