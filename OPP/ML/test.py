from gym import spaces
from collections import deque
from PIL import Image
import numpy as np
import pygame
import cv2
import random


from main_functions import *
from screenshot import *
from resize import *

from model.placement import *
from model.sequence import *


#hyperparameters

epochs = 50000
timestep = 500
learning_rate = 0.001
replay_capacity = 500000
target_update_freq = 10000


# reward hyperparameters max_reward ~ 100
alpha = 9/10*10e-3
beta = 100/40
discount_rate = 0.99








class PackingEnv():
    def __init__(self):
        # make actions closer in value so that the agent experiments with lower actions
        # also give reward for each action taken and make tight timestep so every action counts
        # Action mappings for placement
        self.action_mapping = {
            0: {"name": "left", "values": [1, 5, 10, 25, 50, 100]},
            1: {"name": "right", "values": [1, 5, 10, 25, 50, 100]},
            2: {"name": "up", "values": [1, 5, 10, 25, 50, 100]},
            3: {"name": "down", "values": [1, 5, 10, 25, 50, 100]},
            4: {"name": "rotate_right", "values": [1, 5, 15, 30, 90]},
            5: {"name": "rotate_left", "values": [1, 5, 15, 30, 90]},
            6: {"name": "xreflect", "values": [None]},  # No range
            7: {"name": "yreflect", "values": [None]},  # No range
            8: {"name": "remove", "values": [None]},    # No range
        }

        # 0-8 represents [left, right, up, down, rotate right, xreflect, yreflect, remove, skip? -> rotate left] respectively

        self.replay_buffer = deque(maxlen=replay_capacity)
        self.batch_size = 16


        # initialize
        self.policy_net_seq = network((500, 500, 3))
        self.target_net_seq = network((500, 500, 3))


        self.policy_net_place = UNet((500, 500, 3), 37)
        self.target_net_place = UNet((500, 500, 3), 37)


        self.optimizer_seq = tf.keras.optimizers.Adam(learning_rate=learning_rate) 
        self.optimizer_place = tf.keras.optimizers.Adam(learning_rate=learning_rate) 

        self.place = 0




    def initialize_env(self):
        self.initial_positions = [
    (25.0, 50.0), (75.0, 50.0), (125.0, 50.0), (175.0, 50.0), (225.0, 50.0),
    (275.0, 50.0), (325.0, 50.0), (375.0, 50.0), (425.0, 50.0), (475.0, 50.0),
    (25.0, 125.0), (75.0, 125.0), (125.0, 125.0), (175.0, 125.0), (225.0, 125.0),
    (275.0, 125.0), (325.0, 125.0), (375.0, 125.0), (425.0, 125.0), (475.0, 125.0),
    (25.0, 375.0), (75.0, 375.0), (125.0, 375.0), (175.0, 375.0), (225.0, 375.0),
    (275.0, 375.0), (325.0, 375.0), (375.0, 375.0), (425.0, 375.0), (475.0, 375.0),
    (25.0, 450.0), (75.0, 450.0), (125.0, 450.0), (175.0, 450.0), (225.0, 450.0),
    (275.0, 450.0), (325.0, 450.0), (375.0, 450.0), (425.0, 450.0), (475.0, 450.0)
]
                

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

        # Define constants
        self.SCREEN_WIDTH = 500
        self.SCREEN_HEIGHT = 500
        self.BACKGROUND_COLOR = (0, 0, 0)

        self.BOX_WIDTH = 150
        self.BOX_HEIGHT = 150


        # Set up the display
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        #pygame.display.set_caption('Rotate and Flip Image')

        # shapes_folder is directory that contains images of the shapes
        self.shapes_folder = "shapes"
        self.shape_set = generate_set(self.shapes_folder, num_shapes)

        # Create the sprite group with shape_set, shape_scale, and positions as parameters
        self.shape_scales = random_shape_scale(num_shapes, (0.2)*self.BOX_WIDTH, (0.3)*self.BOX_WIDTH)

        self.shapes = create_sprite_group(self.shape_set, self.shape_scales, self.initial_positions)

        self.delta = 1

        self.currIndex = 0
        self.prevIndex = 0
        
        self.shapes = self.shapes.sprites()

        self.shape_used = self.shapes[self.currIndex]

        # Red box
        self.box_X, self.box_Y = self.SCREEN_WIDTH // 2 - self.BOX_WIDTH // 2, self.SCREEN_HEIGHT // 2 - self.BOX_HEIGHT // 2
        self.box = pygame.Rect(self.box_X, self.box_Y, self.BOX_WIDTH, self.BOX_HEIGHT)
        self.box_mask = pygame.mask.Mask((self.BOX_WIDTH, self.BOX_HEIGHT))
        self.box_mask.fill()


        # Orange box
        self.obox = Shape('Box.png', (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), (self.BOX_WIDTH+3, self.BOX_HEIGHT+3))

        # initialize
        self.counter = 0
        self.touch_object = False




    def render(self):
        pygame.time.delay(1000)

        # manual actions in replay buffer
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()  # Get pressed keys

                # Manual actions via keyboard
                if keys[pygame.K_LEFT]:
                    self.decode_action(2, self.action_mapping)
                if keys[pygame.K_RIGHT]:
                    self.decode_action(8, self.action_mapping)
                if keys[pygame.K_UP]:
                    self.decode_action(14, self.action_mapping)
                if keys[pygame.K_DOWN]:
                    self.decode_action(20, self.action_mapping)
                if keys[pygame.K_r]:
                    self.decode_action(26, self.action_mapping)
                if keys[pygame.K_t]:
                    self.decode_action(31, self.action_mapping)
                if keys[pygame.K_q]:
                    self.decode_action(34, self.action_mapping)
                if keys[pygame.K_e]:
                    self.decode_action(35, self.action_mapping)
                if keys[pygame.K_2]:
                    self.decode_action(36, self.action_mapping)
            else:
                # get the action and value automatically
                self.decode_action(self.place, self.action_mapping)



        if self.counter == 0:
            self.inside_before = False
        
        self.counter+=1
        
        self.shape_used = self.shapes[self.currIndex]
        self.screen.fill(self.BACKGROUND_COLOR)  # Fill the screen with black
        self.screen.blit(self.obox.image, self.obox.changed_pos)
        # Draw box environment
        pygame.draw.rect(self.screen, (255, 255, 255), self.box)


        # Rotate current shape
        self.shape_used.changed_image, self.shape_used.changed_pos = blitRotate(
            self.shape_used.image, (self.shape_used.x, self.shape_used.y),
            (self.shape_used.width // 2, self.shape_used.height // 2), self.shape_used.angle)
        #self.shape_used.changed_width, self.shape_used.changed_height = self.shape_used.changed_image.get_size()

        # Flip current shape
        self.shape_used.changed_image = pygame.transform.flip(self.shape_used.changed_image, self.shape_used.xbool, self.shape_used.ybool)

        # Update mask
        self.shape_used.mask = pygame.mask.from_surface(self.shape_used.changed_image)


        # shape collision
        for i in range(len(self.shapes)):
            if i == self.currIndex:
                continue

            self.screen.blit(self.shapes[i].changed_image, self.shapes[i].changed_pos)

            if self.shapes[i].mask.overlap(
                self.shape_used.mask,
                ((self.shape_used.changed_pos[0] - self.shapes[i].changed_pos[0]),
                (self.shape_used.changed_pos[1]) - (self.shapes[i].changed_pos[1]))):
                self.shape_used.overlap = True
            else:
                self.shape_used.overlap = False

            
        # mask/box collision
        if self.box_mask.overlap(
                self.shapes[i].mask,
            (((self.shape_used.changed_pos[0]) - self.box_X),
            ((self.shape_used.changed_pos[1]) - self.box_Y))):
            touch_box = True
        else:
            touch_box = False

        # outline collision
        if self.obox.mask.overlap(self.shapes[i].mask, (((self.shape_used.changed_pos[0]) - self.obox.changed_pos[0]), ((self.shape_used.changed_pos[1]) - self.obox.changed_pos[1]))):
            self.shape_used.outline = True
        else:
            self.shape_used.outline = True
        

        if touch_box == True and self.shape_used.outline == False:
            self.shape_used.inside = True
        else: 
            self.shape_used.inside = False
        
        self.inside_before = self.shape_used.inside

        if self.prevIndex == self.currIndex:
            self.positions["shape_"+str(self.currIndex)] = (self.shape_used.x, self.shape_used.y)
            self.angles["shape_"+str(self.currIndex)] = self.shape_used.angle
            self.reflections["shape_"+str(self.currIndex)] = (self.shape_used.xbool, self.shape_used.ybool)
            self.inside["shape_"+str(self.currIndex)] = self.shape_used.inside
            self.outline["shape_"+str(self.currIndex)] = self.shape_used.outline
            self.overlap["shape_"+str(self.currIndex)] = self.shape_used.overlap

        self.screen.blit(self.shape_used.changed_image, self.shape_used.changed_pos)

        pygame.display.update()

        


        



    def decode_action(self, flat_index, action_mapping):
        current_index = 0
        self.place = flat_index
        for action_id, action_info in action_mapping.items():
            values = action_info["values"]
            num_values = len(values)
            
            # Check if the flat_index falls within this action's value range
            if current_index <= flat_index < current_index + num_values:
                value_index = flat_index - current_index

                self.delta = values[value_index]

                if action_id == 0:
                    self.shape_used.x -= self.delta
                if action_id == 1:
                    self.shape_used.x += self.delta
                if action_id == 2:
                    self.shape_used.y -= self.delta
                if action_id == 3:
                    self.shape_used.y += self.delta
                if action_id == 4:
                    self.shape_used.angle += self.delta
                if action_id == 5:
                    self.shape_used.angle -= self.delta
                if action_id == 6:
                    self.shape_used.xbool = not self.shape_used.xbool
                if action_id == 7:
                    self.shape_used.ybool = not self.shape_used.ybool
                if action_id == 8:
                    self.shape_used.angle = 0
                    self.shape_used.xbool, self.shape_used.ybool = False, False
                    self.shape_used.x, self.shape_used.y = self.shape_used.original_pos
                    self.inside_before = False

        
            current_index += num_values
        

    def _get_state(self):
        screenshot_window_by_title("pygame window", r"C:\Users\Benson\OPP\ML\screenshot\image.png")


    def _get_obs(self):
        # state: cv2 of the window 
        # observation: positions, rotations, reflections
        return {"shape_positions": self.positions, "shape_angles": self.angles, "shape_reflections": self.reflections}

    def _get_info(self):
        # auxiliary information: empty space inside of box
        imagepath = r"C:\Users\Benson\OPP\ML\screenshot\image.png"
        rImage = cv2.imread(imagepath)
        gray = cv2.cvtColor(rImage, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        area = np.sum(binary == 255)# counts the outline
        return area
    
    def _get_input(self, filename):
        tensor = tf.io.read_file(filename)
        tensor = tf.image.decode_image(tensor, channels=3)
        tensor = tf.image.resize(tensor, [500, 500])
        tensor = tf.expand_dims(tensor, axis=0)
        tensor = tf.cast(tensor, tf.float32) /255.0                # Normalize
        return tensor
    



    def step(self):
        

        self.reward = 0 # fix this so it doesnt keep checking everytime
        # read pixelwise environment and run it through the model
        if self.counter == 0:
            self._get_state()
            #self.resize_img()
            self.state = self._get_input(r"C:\Users\Benson\OPP\ML\screenshot\image.png")
        else: 
            self.state = self.state_

        info = self._get_info()

        #self.state = self.state[1:]
        #self.state = tf.zeros(tf.shape(self.state), dtype=tf.float32)

        self.seq = np.argmax(self.policy_net_seq(self.state))

        self.place = np.argmax(self.policy_net_place(self.state))

        print(self.seq, self.place)



        
        # sequence
        self.currIndex = self.seq

        # run in env
        self.render()

        self.currIndex = self.prevIndex

        # reward
        num_inside = sum(1 for v in self.inside.values() if v is True)
        a = (22500-info) * alpha
        b = num_inside * beta
        self.reward += (a + b)
        # a>b


        # next state
        self._get_state()
        #self.resize_img()
        self.state_ = self._get_input(r"C:\Users\Benson\OPP\ML\screenshot\image.png")
        
        
        print(f"Total reward: {self.reward:.0f}")



    def reset(self):
        # get observation and info then call main

        self.counter = 0
        self.reward = 0
        # reset reward
        
        self.render()
        



    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            cv2.destroyAllWindows()



env = PackingEnv()

pygame.init()

'''
try:
    load_weights(env.policy_net_seq)
    load_weights_UNET(env.policy_net_place)
    print("weights have been successfully loaded")
except:
    pass
'''

for episode in range(epochs):
    # intialize starting state

    env.initialize_env()
    env.reset()
    save_weights(env.policy_net_seq)
    save_weights_UNET(env.policy_net_place)
    # for reset
    env.counter -= 1

    # sparse rewards
    if any(env.overlap.values()): 
        # - max(a+b) * self.discount_rate * 1/num_objects = -(max(a) + max(b))*self.discount_rate*1/num_objects
        env.reward -= (((40*beta) + (90000*alpha))* discount_rate * 1/40)
        #discount_rate = discount_rate*discount_rate
        env.touch_object = False
    
    if any(env.outline.values()):
        env.reward -= ((40*beta) + (90000*alpha))

    for _ in range(timestep):
        print("Timestep: ", _)
        env.step()




    
    