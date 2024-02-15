import numpy as np
import pygame
from matplotlib import pyplot as plt

def sigmoid(z):
     return 1/(1+np.exp(-z))
def RELU(z):
     return (z+np.abs(-z))/2

def mutate(weight):
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            if  np.random.rand() < 0.02:
                weight[i, j] += np.random.rand() * 10 - 5
    return weight

POP_A = 120
POP_B = 60
WIDTH = 1200
HEIGHT = 675

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Generation Algorithm")
clock = pygame.time.Clock()


class Nutrient(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((5, 5))
        self.image.fill((170, 170, 90))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
    
class Sensor(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10, 10))
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
    def Kill(self):
        self.kill()

class Cell_A(pygame.sprite.Sprite):
    def __init__(self, x, y, weight1, weight2):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10, 10))
        self.image.fill((90, 170, 90))    
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.weight1 = weight1
        self.weight2 = weight2
        self.degree = np.random.randint(0,360)
        self.energy = 20
               
    def network(self):
        self.input_layer = np.zeros((1,25))
        
        for j in range(25):
            for i in range(5):
                sensor_front = Sensor(self.rect.center[0] + (15 + 20 * i) * np.cos(((self.degree - 90 + 7.5 * j) / 180) * 3.14159),
                                      self.rect.center[1] + (15 + 20 * i) * np.sin(((self.degree - 90 + 7.5 * j) / 180) * 3.14159))
            
                if len(pygame.sprite.spritecollide(sensor_front, cell_B, False)) != 0:
                    self.input_layer[0, j] = 1 + i
                    sensor_front.Kill()
                    break
                sensor_front.Kill()
                
        
        return np.tanh(sigmoid(self.input_layer.dot(self.weight1)).dot(self.weight2))
         
    def update(self):
        
        # for j in range(25):
        #     for i in range(5):
        #         self.obstacle_hitbox_front = pygame.Rect(0, 0, 10 , 10)
        #         self.obstacle_hitbox_front.center = (self.rect.center[0] + (15 + 20 * i) * np.cos(((self.degree - 90 + 7.5 * j) / 180) * 3.14159),
        #                                              self.rect.center[1] + (15 + 20 * i) * np.sin(((self.degree - 90 + 7.5 * j) / 180) * 3.14159))
        #         pygame.draw.rect(screen, (255, 150, 150), self.obstacle_hitbox_front)    
        
  
        
        self.output_layer = self.network()
        self.speed = (self.output_layer[0, 1]) * 4
        self.degree += self.output_layer[0, 0] * 5
        self.rect.x += self.speed * np.cos((self.degree / 180) * 3.14159)
        self.rect.y += self.speed * np.sin((self.degree / 180) * 3.14159)

        if self.rect.x > WIDTH or self.rect.x < 0:
            self.rect.x -= self.speed * np.cos((self.degree / 180) * 3.14159)
            self.degree = 180-self.degree
        if self.rect.y > HEIGHT or self.rect.y < 0:
            self.rect.y -= self.speed * np.sin((self.degree / 180) * 3.14159)
            self.degree = -self.degree
        
        self.energy -= 0.05 + np.abs(self.speed) / 500
        if self.energy <= 0:
            self.kill()
            
        elif self.energy >= 40:
            self.energy = 20
            weight1 = mutate(self.weight1)
            weight2 = mutate(self.weight2)
            cell = Cell_A(self.rect.x, self.rect.y, weight1, weight2)
            cell_A.add(cell)

    def eat(self):
        self.energy += 5

class Cell_B(pygame.sprite.Sprite):
    def __init__(self, x, y, weight1, weight2):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10, 10))
        self.image.fill((170, 90, 90))  
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.weight1 = weight1
        self.weight2 = weight2
        self.degree = np.random.randint(0,360)
        self.energy = 50
       
    def network(self):
        self.input_layer = np.zeros((1,13))
        
        for j in range(13):
            for i in range(5):
                sensor_front = Sensor(self.rect.center[0] + (15 + 20 * i) * np.cos(((self.degree - 30 + 5 * j) / 180) * 3.14159),
                                      self.rect.center[1] + (15 + 20 * i) * np.sin(((self.degree - 30 + 5 * j) / 180) * 3.14159))
            
                if len(pygame.sprite.spritecollide(sensor_front, cell_B, False)) != 0:
                    self.input_layer[0, j] = 1 + i
                    sensor_front.Kill()
                    break
                sensor_front.Kill()
                
        
        return np.tanh(sigmoid(self.input_layer.dot(self.weight1)).dot(self.weight2))               
    
    def update(self):
        
            
        # for j in range(13):
        #     for i in range(5):
        #         self.obstacle_hitbox_front = pygame.Rect(0, 0, 10 , 10)
        #         self.obstacle_hitbox_front.center = (self.rect.center[0] + (15 + 20 * i) * np.cos(((self.degree - 30 + 5 * j) / 180) * 3.14159),
        #                                              self.rect.center[1] + (15 + 20 * i) * np.sin(((self.degree - 30 + 5 * j) / 180) * 3.14159))
        #         pygame.draw.rect(screen, (255, 150, 150), self.obstacle_hitbox_front)    
      
        
        self.output_layer = self.network()
        self.size = 1.5
        self.speed = 3 + (self.output_layer[0, 1]) * 2
        self.degree += self.output_layer[0, 0] * 5
        self.rect.x += self.speed * np.cos((self.degree / 180) * 3.14159)
        self.rect.y += self.speed * np.sin((self.degree / 180) * 3.14159)

        if self.rect.x > WIDTH or self.rect.x < 0:
            self.rect.x -= self.speed * np.cos((self.degree / 180) * 3.14159)
            self.degree = 180-self.degree
        if self.rect.y > HEIGHT or self.rect.y < 0:
            self.rect.y -= self.speed * np.sin((self.degree / 180) * 3.14159)
            self.degree = -self.degree
        
        self.energy -= 0.15 + np.abs(self.speed) / 300
        if self.energy <= 0:
            self.kill()
            
        elif self.energy >= 100:
            self.energy = 50
            weight1 = mutate(self.weight1)
            weight2 = mutate(self.weight2)
            cell = Cell_B(self.rect.x, self.rect.y, weight1, weight2)
            cell_B.add(cell)

    def eat(self):
        self.energy += 25
        self.degree += 180
          
cell_A = pygame.sprite.Group()
cell_B = pygame.sprite.Group()
nutrient = pygame.sprite.Group()

for i in range(POP_B):    
    weight1 = np.random.rand(13, 8) * 10 - 5
    weight2 = np.random.rand(8, 2) * 10 - 5
    cell = Cell_B(np.random.randint(5, WIDTH-5), np.random.randint(5, HEIGHT-5), weight1, weight2)
    cell_B.add(cell)
for i in range(POP_A):
    weight1 = np.random.rand(25, 8) * 10 - 5
    weight2 = np.random.rand(8, 2) * 10 - 5
    cell = Cell_A(np.random.randint(5, WIDTH-5), np.random.randint(5, HEIGHT-5), weight1, weight2)
    cell_A.add(cell)


running = True

list1 = []
list2 = []
list3 = []
time = 0

while running:
    clock.tick(100)
    time += 0.01
    for i in range(2):
        cell = Nutrient(np.random.randint(5, WIDTH-5), np.random.randint(5, HEIGHT-5))
        nutrient.add(cell)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running =False
    
    screen.fill((198,198,200))
    

    nutrient.draw(screen)
    cell_A.draw(screen)
    cell_B.draw(screen)
    cell_A.update()
    cell_B.update()
    hits = pygame.sprite.groupcollide(cell_B, cell_A, False, True)
    for hit in hits:
        hit.eat()
    
    hits = pygame.sprite.groupcollide(cell_A, nutrient, False, True)
    for hit in hits:
        hit.eat()
    list1.append(time)
    list2.append(len(cell_A))
    list3.append(len(cell_B))
    print("Time:", round(time, 2), "s   cell A population:", len(cell_A),"   cell B population:", len(cell_B))
    pygame.display.update()
pygame.quit() 
plt.plot([],[],color='m', label='cell A population', linewidth=5)
plt.plot([],[],color='c', label='cell B population', linewidth=5)
plt.stackplot(list1, list2,list3, colors=['m','c'])

plt.xlabel('time')
plt.ylabel('population')

plt.legend()
plt.show()
plt.plot(list1, list2,
         list1, list3)
plt.show()