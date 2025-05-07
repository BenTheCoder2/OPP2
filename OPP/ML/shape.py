import pygame


class Shape(pygame.sprite.Sprite):
    def __init__(self, img_file, position, scale):
        pygame.sprite.Sprite.__init__(self)
        self.original_image = pygame.image.load(img_file).convert_alpha()
        self.image = pygame.transform.scale(self.original_image, scale)
        self.rect = self.image.get_rect(topleft=position)
        self.changed_image = self.image
        self.width, self.height = self.image.get_size()

        
        self.changed_width, self.changed_height = self.width, self.height
        self.original_pos = position
        self.x = position[0] # is this the top left?
        self.y = position[1]
        self.angle = 0
        self.xbool = False
        self.ybool = False
        self.rem = False
        self.changed_pos = (self.x - self.width // 2, self.y - self.height // 2)

        # Create collision mask from the image
        self.mask = pygame.mask.from_surface(self.image)


# Create sprite group to contain shapes
def create_sprite_group(shape_set, scale, positions):
    #positions gives position of centers not top left
    
    shapes = pygame.sprite.Group()

    for i in range(len(shape_set)):
        shapes.add(Shape(shape_set[i], positions[i], scale))


    return shapes