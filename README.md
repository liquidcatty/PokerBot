# PokerBot

## Core of project
Make a strong bot for games of no-limit Texas Hold'em poker. The model for playing will be wrapped in a UI to give advice for your games of poker through a simple website.

## Project is divided into 2 parts:

### 1. Creating and training the model
For this task I plan to use a transformer. The model will be trained through self-play and reinforcement learning.  

#### What is RL?  
Reinforcement learning is based around the idea that we have 3 things:  
1. Model  
2. Environment  
3. Reward from the environment  

What is done is we put the model in the environment, get a reward from the environment which shows the performance of the model, and train the model based on this value. That's RL in short.  

#### How it is done in this project  
As you probably know, poker is a very random game. It has a lot of noise for the model to learn from just one game. So how to find better players? Instead of the environment being just one poker game, it will be a handful of them. Thanks to this, the best model can be picked (at least the chance is higher). Also, each game is played through self-play. As you can guess, the model plays with copies of itself to pick the best one. During training, this process is repeated each epoch.  

#### What model should be used?  
Poker is, in its core, a sequential game. So a common intuition is to use a transformer, because it is made for tasks like that. Right now, I'm settling for using a single transformer for this, but I expect that once the first implementation is done, I will experiment with other ways of creating the architecture. Ideas I have right now are, for example, adding more algorithmic parts to it inspired by Pluribus (one of the more powerful bots out there), or dividing one transformer into separate transformers for each thought process of the model.  

### 2. How to make UI
I don't have many exact ideas about this part yet, probably simple input/output packed in nice graphics. I plan for the website to run on a server, because I don't want to fry anyone's older device by running my model on it (if it's light enough, maybe it could run on the user's device).