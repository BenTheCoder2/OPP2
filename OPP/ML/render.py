



self.initial_positions = [(50, 100), (150, 100), (250, 100), (350, 100), (450, 100),
                (550, 100), (650, 100), (750, 100), (850, 100), (950, 100),
                (50, 250), (150, 250), (250, 250), (350, 250), (450, 250),
                (550, 250), (650, 250), (750, 250), (850, 250), (950, 250),
                (50, 750), (150, 750), (250, 750), (350, 750), (450, 750),
                (550, 750), (650, 750), (750, 750), (850, 750), (950, 750),
                (50, 900), (150, 900), (250, 900), (350, 900), (450, 900),
                (550, 900), (650, 900), (750, 900), (850, 900), (950, 900)]
    

num_shapes = 40

# Create keys once
shape_keys = [f"shape_{i}" for i in range(num_shapes)]

# Positions based on initial positions
self.positions = {key: self.initial_positions[i] for i, key in enumerate(shape_keys)}

# All other attributes with default values
self.angles = {key: 0 for key in shape_keys}
self.reflections = {key: (False, False) for key in shape_keys}
self.inside = {key: False for key in shape_keys}
self.outline = {key: False for key in shape_keys}
self.overlap = {key: False for key in shape_keys}




def main():
    

    # Define constants
    SCREEN_WIDTH = 1000
    SCREEN_HEIGHT = 1000
    BACKGROUND_COLOR = (0, 0, 0)

    BOX_WIDTH = 300
    BOX_HEIGHT = 300


    # Set up the display
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    #pygame.display.set_caption('Rotate and Flip Image')

    # shapes_folder is directory that contains images of the shapes
    shapes_folder = "shapes"
    shape_set = generate_set(shapes_folder, 40)

    # Create the sprite group with shape_set, shape_scale, and positions as parameters
    shape_scales = random_shape_scale(40, (0.2)*BOX_WIDTH, (0.3)*BOX_WIDTH)

    shapes = create_sprite_group(shape_set, shape_scales, positions)

    delta = 10

    currIndex = 0
    prevIndex = 0

    shapes = shapes.sprites()

    shape_used = shapes[currIndex]

    # Red box
    box_X, box_Y = SCREEN_WIDTH // 2 - BOX_WIDTH // 2, SCREEN_HEIGHT // 2 - BOX_HEIGHT // 2
    box = pygame.Rect(box_X, box_Y, BOX_WIDTH, BOX_HEIGHT)
    box_mask = pygame.mask.Mask((BOX_WIDTH, BOX_HEIGHT))
    box_mask.fill()

    # Orange box
    obox = Shape('Box.png', (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2), (BOX_WIDTH+5, BOX_HEIGHT+5))


    run = True
    counter = 0



    while run:
        pygame.time.delay(100)

        if counter == 0:
            inside_before = False
        
        counter+=1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False



        #inputs
        keys = pygame.key.get_pressed()
        if keys[pygame.K_1]:
            currIndex += 1
            currIndex %= len(shapes)
            inside_before = False
            # observation
            self.positions["shape_"+str(self.currIndex)] = (shape_used.x, shape_used.y)
            self.angles["shape_"+str(self.currIndex)] = shape_used.angle
            self.reflections["shape_"+str(self.currIndex)] = (shape_used.xbool, shape_used.ybool)

            print(insideList)
        else:
            if keys[pygame.K_LEFT]:
                shape_used.x -= delta
            if keys[pygame.K_RIGHT]:
                shape_used.x += delta
            if keys[pygame.K_UP]:
                shape_used.y -= delta
            if keys[pygame.K_DOWN]:
                shape_used.y += delta
            if keys[pygame.K_r]:
                shape_used.angle += delta
            if keys[pygame.K_t]:
                shape_used.angle -= delta
            if keys[pygame.K_q]:
                shape_used.xbool = not shape_used.xbool
            if keys[pygame.K_e]:
                shape_used.ybool = not shape_used.ybool
            if keys[pygame.K_2]:
                shape_used.angle = 0
                shape_used.xbool, shape_used.ybool = False, False
                shape_used.x, shape_used.y = shape_used.original_pos
                inside_before = False
            if keys[pygame.K_SPACE]:
                main()
        
                



        shape_used = shapes[currIndex]
        screen.fill(BACKGROUND_COLOR)  # Fill the screen with black
        screen.blit(obox.image, obox.changed_pos)
        # Draw box environment
        pygame.draw.rect(screen, (255, 255, 255), box)


        # Rotate current shape
        shape_used.changed_image, shape_used.changed_pos = blitRotate(
            shape_used.image, (shape_used.x, shape_used.y),
            (shape_used.width // 2, shape_used.height // 2), shape_used.angle)
        #shape_used.changed_width, shape_used.changed_height = shape_used.changed_image.get_size()

        # Flip current shape
        shape_used.changed_image = pygame.transform.flip(shape_used.changed_image, shape_used.xbool, shape_used.ybool)

        # Update mask
        shape_used.mask = pygame.mask.from_surface(shape_used.changed_image)


        # shape collision
        for i in range(len(shapes)):
            if i != currIndex:

                screen.blit(shapes[i].changed_image, shapes[i].changed_pos)

                if shapes[i].mask.overlap(
                    shape_used.mask,
                    ((shape_used.changed_pos[0] - shapes[i].changed_pos[0]),
                    (shape_used.changed_pos[1]) - (shapes[i].changed_pos[1]))):
                    continue

            
            # mask/box collision
            if box_mask.overlap(
                    shapes[i].mask,
                (((shape_used.changed_pos[0]) - box_X),
                ((shape_used.changed_pos[1]) - box_Y))):
                touch_box = True
            else:
                touch_box = False

            # outline collision
            if obox.mask.overlap(shapes[i].mask, (((shape_used.changed_pos[0]) - obox.changed_pos[0]), ((shape_used.changed_pos[1]) - obox.changed_pos[1]))):
                touch_outline  = True
            else:
                touch_outline = False
            

            if touch_box == True and touch_outline == False:
                inside = True
            else: 
                inside = False
            
            # if touches outline and inside box
            if inside_before  == True and touch_outline == True:
                continue
            
            inside_before = inside

            insideList["shape_"+str(currIndex)] = inside

               

        screen.blit(shape_used.changed_image, shape_used.changed_pos)

        pygame.display.update()
    pygame.quit()



main()