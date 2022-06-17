---
layout: post
title: "Learning Rate Schedulers"
date: 2022-06-11
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Learning Rate Schedulers

Relevant docs:
- [Learning rate schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

Hello, everyone, and welcome to the PyTorch podcast.
Today, I want to talk about learning rate schedulers on request of Nelson Elhage.

## Learning rates
What is a learning rate? Well, remember, deep learning is all about optimization and optimization is all about, you know, starting off at some point in your very hyper dimensional parameter space.
and then slowly making your way to a set of parameters which does better.
And so every step we do, you know, is based off of the gradient we compute for computation.
And so the learning rate simply says, you know, once I'm at some point, in my primary space and I figure out where I wanna go (my gradient) how far do I go in that direction before I stop and, you know, reassess the reassess the landscape and, you know, compute my gradient again and go further.
So that is the learning rate.

So why does the learning rate matter? Well, you can think about the situation and if you have a very, you know, spiky landscape where there's a lot of different changes to the gradient, then if you do a very large learning rate and you make a very large step when you're doing an optimization, you might sort of bypass so you might may have been, like, locally improving the loss you know, for a very small amount of the step.
But then, you know, the landscape changed and now you're climbing back up the hill and you just went too far and you overshot the place you wanted to go.

A very common diagram and sorry, this is a podcast, so I can't show you a picture.[^lrozzilation]
Imagine you have some sort of valley where the valley sort of is slowly going down until you get to the, you know, global optimum.
In this case, we'll have the global optimum be something that's low because this graph is representing our loss, so the lower the losses, the better.
So if you start your ball, the ball being, you know, the point we are at on the parameter space, On the side of the valley, then if you go too far, you will bypass the sort of the the bottom the deepest point of the valley and hop to the other side of the valley and then you'll sort of zigzag back and forth until eventually you get to the final destination, but you'll do a lot of wasted steps along the way.
So, you know, a lot of sort of optimization techniques and a lot of playing around with learning rate.
It's all about sort of trying to get to your final destination, you know, more directly without, you know, overshooting every single time.
That being said, you don't want your learning rate to be too small either because, well, if it's a really small learning rate, then you're just not making very much progress at any given seven times.
So, you know, if you don't make very much progress, you might just never actually get to convergence on your network in this situation.

So learning rates are kind of important and, you know, certainly, when you are writing a published model, you'll have some sort of optimizer.
And your optimizer is gonna make some decisions about how exactly it's going to go about go about exploring the state space. 
Most optimizers have a hyper parameter called the learning rate, which is just a global number you can toggle to say instead of how, you know, far or close you should go.
There are some optimizers that automatically determine a good learning rate, but there are also optimizers which don't.
And so that's just a parameter too.

## Learning rate schedulers
So a learning rate scheduler is a way to sort of automatically modify this hyper parameter on your optimizers in some way that's sort of non standard.
Right? Because there's a lot of things you might wanna do.
Maybe while you are warming up, you know, while you're doing the initial, you know, few steps of your computation, you don't want to, you know, go too far.
So you wanna sort of just slowly explore your local space ramping up until you actually hit your final learning rate.
Or maybe, you know, as time goes on, you want your learning rate to decay and get smaller and smaller so that after you've done all the major learning at the beginning, you're gonna finally get closer and close to your the final optimum.
And now you wanna make smaller and smaller steps so you are careful not to overshoot in the situation.
So there are tons and tons of, you know, different learning rate schedulers.
Honestly, there aren't that many So if you're like, look at [`torch.optim`](https://pytorch.org/docs/stable/optim.html), that's the directory that has all of our optimizers.
We have tons and tons of optimizers because there are lots of, you know, ways to go over doing optimizations.
Our learning rate schedulers fit in a [single Python file](https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py).
So, you know, it's really not there's not that much stuff going on there.
But it's something that, you know, people do care about, people do use, and that's what I wanna talk about in the podcast today.

### Stateful LR schedulers
So where I wanna go next is how exactly does the learning rate scheduler API in PyTorch work?
It's kind of surprising.
And we basically haven't changed it since, you know, the very beginning, I think it was, like, `v0.1`.
Someone submitted a pull request to add earning rate schedulers.
to PyTorch and we were like, okay, we'll add it.
And we have basically not changed the API since then.
A lot of APIs in PyTorch's neural network, you know, library and Python have not changed.
So this can result in some weirdness in the API, things we learn over time and we haven't been able to fix them.
Well, let's talk a little bit about what this learning rate schedule API looks like.
So the learning rate scheduler API is sort of based off of two things.

The first thing it's based off of is it's based off the optimizer API.
Why is this important?
Well, optimizers in PyTorch are stateful.
So the standard model for, you know, PyTorch program is you know, you're you're off doing your optimization.
And the way things work is you go ahead, you run your computation, you run your forwards and backwards, you compute the gradients, and then you ask the optimizer to do a step.
And the step, you know, is a method on the optimizer.
It's a stateful method that has side effects.
And what it does is it goes ahead and reads all the gradients, updates the optimizer's internal state, and updates all the parameters to actually, you know, make them all go well.
So, you know, the model that people have is, you know, they're looping through their batches of inputs.
And every batch they do, they call the optimizer step at the end.

So our intrepid contributor back in the day looked at this API and then, like, okay.
Well, let's do something similar for learning rate scheduling.
So What they did was they said, okay.
We'll have an API for, you know, modifying the learning rate.
It's gonna be a learning rate scheduler object and you will call step on it to modify the learning rate.
Unlike optimizers, you know, optimization is happening every minibatch.
Right? Because, you know, every batch you do, you want to actually update the parameters with what you did.
Typically, for a learning rate, you only wanna do that for entire epoch.
You don't wanna modify the learning rate until you've actually finished processing the entirety of your input data set.
So learning rate has to have its own step function, but okay, fine.
So, you know, you have your optimizer step.
You have your learning rate step.
And, you know, you just call them when appropriate, either at the end of your training iteration or at the end of your training epoch.

That being said, actually, in the beginning, learning rate schedulers were implemented in kind of funny way.
And whether or not you call them before or the optimizer step or after the optimizer step was something that sort of wasn't well specified.
So we had to we had to make a busy breaking change to sort of fix it so that, you know, the behavior was uniform.
You always call it after the optimizer set.
It was pretty confusing because, you know, stateful APIs are confusing.
It's hard to, you know, make sure that they do exactly the right thing.
And, you know, you you don't even notice half the time.
Right? because the learning rate is just this hyper parameter and obviously your optimization is still gonna work even if you stay stuck on your old learning rate you know, one epoch more than you accept it.
So the the kind of people who notice this sort of thing is, like, if they have some network super sensitive to the initial conditions or maybe they're trying to reproduce a paper and they're like, how come the learning rate is not the same thing as, you know, what I saw in the paper and, well, that's because, you know, we messed up the stateful API.
That's simple.

So I mentioned that the learning rate API was based off of this optimizer stateful API.
Right? So it's like you say, okay.
You know, when I'm done, I run step and that will update the learning rates everywhere.
But the second thing, and this is important, is that PyTorch's learning rate schedulers were essentially cribbed from Keras.
So, you know, Keras was, you know, existed back then, and Keras has been around for a while.
And Keras had a learning rate learning rate scheduler API.
And, basically, besides, you know, state statefulizing up the API.
because Keras's learning rate API is sort of based on a sort of callback model where you're, you know, the optimizer calls into the learning rate, callback to figure out what to do.
Basically, crafted into the stateful API, but using the same algorithms that Keras was using to determine learning rate.
And in particular, the way Keras computes learning rate is you are at some point in your computation and you are, you know, you basically are at some epoch, you know, epoch 10, epoch 20, whatever.
And you have a formula which says, given this epoch, what should my learning rate be? So this is a closed form formula.
It, you know, takes in the epoch, produces learning rate, and that's what you set everything to.

So we've got the stateful API, but what the stateful API is doing under the hood is it's just going ahead and running this closed form compute to figure out what the next step should be.
So actually, there is no there's no statefulness beyond the fact that, you know, you just call step in this internal state mutates.
Actually, this is why the step function for the longest time accepted an epoch parameter, and you could use this to sort of, you know, time travel your learning rate schedulers into the future by Ep 1, Ep 2, Ep 100.
Whatever, you know, that's fine.
It's gonna work.
Why does it work? It's because there's a closed form formula, right, and we can just zoom straight to that.
spot.
And, you know, that seemed reasonable-ish.

### Compositional learning rate schedulers
The problem with giving people stateful APIs is they start looking at what the stateful APIs do, and they start expecting them to actually be stateful.
So pretty early in PyTorch's life, we got a feature request.
And the feature request was I'd like to have so called chainable or as I like to say, composable learning rate schedulers.
So the ask here was, you know, sometimes people wanna combine various learning rate strategies.
Right? They might have a a learning rate strategy where they are doing, you know, they're doing some sort of decay as the training run goes on and on and on, but they want some special behavior at the very beginning.
And so they'll have an extra learning rate scheduler just for handling that sort of situation.
And it's not really obvious how to mash together two learning rate schedulers.
Certainly, if they're using the closed form solutions, they're just not compositional at all because let's say that you have one learning rate scheduler and you call it, and it figures out, oh, hey, the learning rate should be 5 at at this epoch.
And it sets all the learning rates to 5.
And then the next learning rate scheduler, you know, says, okay.
Well, this is the current epoch and the learning schedule.
should be eight, and then it sets 8 to everyone.
And actually, you know, like, they basically don't communicate with each other at all.
Right? The closed form solutions are actually not compositional in this way.

So people were like, hey, you know, it would be cool if, you know, actually if I, you know, did a learning rate schedule step and then another learning rate schedule step it actually did what the API suggested.
That is to say, you know, we've got the stateful API, a step should, you know, transform the learning rate to the next learning rate.
And, like, you know, that's what I expected to do.
And through the efforts of Chandler Zoe and then later Vincent Quinnville Blair, we actually did exactly that.
We took all of the closed form formulas that were previously in the learning rate scheduler and we essentially figured out how to turn them into the single step functions that would give you the same result as the closed form solution.
So now you could actually compose these things because you would say, okay.
Well, first, I apply the step implied by the first learning rate schedule, and then I apply the step implied by the second learning rate schedule.
and now you actually have composational learning rate schedulers.
Wahoo.

Well, actually the first time we try to land is it broke because remember you've got time traveling epochs.
And if you're gonna do time traveling epochs, I don't know how you're gonna do the stepping thing because, like, does that even work? 
You don't have a closed form solution anymore.
And so basically, you only have a choice of either going ahead and playing out the, you know, epochs one by one if you have the time traveling epoch or you can do what we actually did, which is we've got a closed form solution in our back pocket.
It's like it's like an underscore method closed form.
and we just call that and we're like, oh, it's not gonna be compositional if you're time traveling.

Alright.
So the reason why I did this podcast episode is because essentially Nelson came to me and was like, hey, what the heck is going on with these learning grade schedulers? Like, it it feels like someone you know, had a dare to make it as stateful as possible, and they followed through with a dare.
And the answer to that is yes.
That is basically what happened.
Right? We started off with a stateful API wrapping over a functional closed form computation of learning rate schedules.
But people were like, hey, you know, stateful API.
I'm expecting it to be stateful.
And so we turned the insights into the stateful version.

Was this the right decision? I have no idea.
I managed to trick several people into, you know, making this possible.
So if it was a bad decision, I suppose it wasn't obviously a bad decision.
But with the benefit of hindsight, I'm not really sure I would have gone about doing it the same way.
Probably the distinction that we probably should have made is there are some learning rate schedules that are compositional and some that are not.
Right? So like if I'm going to do a exponential learning rate, And then I want to compose this with something that sort of fiddles around with the initial conditions of my learning rate.
What I'm probably expecting to happen is I start off with my exponential rate, you know, exactly as is.
And then I'm just gonna, you know, do a transformation on that learning rate afterwards.
And so probably people weren't expecting to, like, arbitrarily compose, you know, an exponential learning rate with a you know, step LR learning rate, you know, all sorts of random compositions.
Probably, that's not actually what people wanna do.
They probably only want a set of compositional ones, but then the basic learning rate schedules, those probably just wanna be closed form.
Maybe.
I have no idea.

One of the things about learning rate schedules in PyTorch is, as I said, it is very simple.
The API is not so simple.
sorry.
We're kinda stuck with the stateful API, but it's very easy to write your own learning rate scheduler.
And so, you know, with a lot of things in PyTorch, you know, core library, sometimes they're just not very well put together.
And it's been okay.
It's because people can just write their own, you know, schedulers and do their own thing.
And that's always been, you know, one of the things about PyTorch that, you know, hey, if there's some piece you don't like.
Well, this is just a library.
You don't have to use it.
You can know, write your own thing.
And really all the learning rate scheduler is doing is it's going into the optimizers and just updating their internal learning So you you can absolutely as I said, it's just one file.
You can go ahead and do your own thing.
And people have gone ahead and done their own thing.
I know for the very least, like, [Classy Vision](https://classyvision.ai/) had their own, you know, implementation of learning rate schedules from scratch.
So that's learning rate schedulers.

## Statefullness vs Functional APIs
I mean, I talked about what learning rates are and what really is going on here, I think, is just a question about Pytorch's API design.
Right? One of the things that made Pytorch really, really successful was that we let people work with `nn.module`s in a imperative, mutable way.
It's just very, very natural for people.
If you look in the JAX's world, people are, you know, trying to discover how to make neural networks work with a functional API where you don't have stateful operations.
That's, you know, pretty interesting.
I think they've come up with some pretty good stuff, but it's also non obvious what exactly that API should be because it's just, like, just less natural for people.
And so as a result, there's lots of libraries exploring different corners of the design space.
I actually think they will probably figure out a really good design in the end, but it's gonna take, you know, a dozen libraries to get there.
Or, I guess, we can be in the Pytorch world where it's like, hey, mutation everywhere, hooray.
And, you know, Also, it's very very complicated.
It's probably more complicated than the functional API.
But, I mean, people seem to like it.
So who am I to quibble with them?
This is me, right, a formerly rabid purely functional programmer.
I used to work on the GHC Haskell compiler.
And now I'm like, hey, you know, mutation is great.
I just use all of my functional programming tricks to help reason about what the code is supposed to do in the end.

Alright.
That's everything I wanted to talk about today.
Talk to you next time.

[^lrozzilation]: See for example [this blogpost](https://www.jeremyjordan.me/nn-learning-rate/) for the kind of picture Edward is talking about here.
