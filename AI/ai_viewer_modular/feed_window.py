import pygame
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from .constants import PTZ_COMMANDS
import requests

class FeedWindow:
    def __init__(self, name, camera_feed, x, y, width=320, height=240):
        self.name = name
        self.feed = camera_feed
        self.rect = pygame.Rect(x, y, width, height)
        self.dragging = False
        self.resizing = False
        self.selected = False
        self.resize_margin = 10

    def draw(self, surface, font):
        composite = self.feed.get_composite()
        if composite is None:
            return
        frame_rgb, overlay_rgba = composite
        frame_rgb = cv2.resize(frame_rgb, (self.rect.width, self.rect.height))
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
        surface.blit(frame_surface, (self.rect.x, self.rect.y))
        if overlay_rgba is not None:
            overlay_rgba = cv2.resize(overlay_rgba, (self.rect.width, self.rect.height))
            overlay_surface = pygame.image.frombuffer(overlay_rgba.tobytes(), overlay_rgba.shape[1::-1], "RGBA")
            surface.blit(overlay_surface, (self.rect.x, self.rect.y))
        pygame.draw.rect(surface, (0, 255, 0) if self.selected else (255, 255, 0), self.rect, 2)
        surface.blit(font.render(self.name, True, (255, 255, 255)), (self.rect.x + 5, self.rect.y + 5))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3 and self.rect.collidepoint(event.pos):
            root = tk.Tk()
            root.withdraw()
            selection = simpledialog.askstring("Overlay Options", "Options: trails, ids, speed")
            if selection:
                if "trail" in selection.lower():
                    self.feed.show_trails = not self.feed.show_trails
                if "id" in selection.lower():
                    self.feed.show_ids = not self.feed.show_ids
                if "speed" in selection.lower():
                    self.feed.show_speed = not self.feed.show_speed
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self._in_resize_zone(event.pos):
                self.resizing = True
                self.selected = True
            elif self.rect.collidepoint(event.pos):
                self.dragging = True
                self.mouse_offset = (self.rect.x - event.pos[0], self.rect.y - event.pos[1])
                self.selected = True
            else:
                self.selected = False
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
            self.resizing = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self.rect.x = event.pos[0] + self.mouse_offset[0]
                self.rect.y = event.pos[1] + self.mouse_offset[1]
            elif self.resizing:
                self.rect.width = max(100, event.pos[0] - self.rect.x)
                self.rect.height = max(80, event.pos[1] - self.rect.y)

    def _in_resize_zone(self, pos):
        x, y = pos
        return self.rect.collidepoint(x, y) and (
            abs(x - (self.rect.x + self.rect.width)) < self.resize_margin or
            abs(y - (self.rect.y + self.rect.height)) < self.resize_margin)
