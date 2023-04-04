# Cognitive Robotics Project - Kaitei Takara Sagashi
Lior Avrams, Bar Hacohen, Michael Rosenblum, and Roy Waisbord

# Introduction 

This project simulates a game with a submarine agent in an underwater environment, whose task is to find treasures and return them to a boat on the surface. The solution to this game is a compound implementation of a searching algorithm we created and an MDP solving algorithm, policy iteration. Its purpose is to showcase a possible approach to solving a POMDP problem by reducing it first to an MDP, then using the simpler and less computationally expensive solution. 

Remark: For a map with two-three treasures, the solution may take up to 20 seconds to solve the problem after seeing the treasure (patience is needed since it’s not the most efficient solution) 

# Installation/Setup 

The user should have numpy, matplotlib, and pygame installed. 

For this project we used the “Pygame” library to animate the problem. The library can be downloaded from the Visual Studio extensions tab and is required to run the code properly.  

If the user has “pip” installed, running the following command in the terminal will solve any dependency issues: 

```
pip install numpy matplotlib pygame 
```

# Usage 

To run this project, download the .zip file and extract the folder. Navigate to this folder in the command line and run the following command: 

```
python3 main.py
```
