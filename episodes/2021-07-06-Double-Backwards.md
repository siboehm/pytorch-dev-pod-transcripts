---
layout: post
title: "Double Backwards"
date: 2021-07-06
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Double Backwards

Hello, everyone, and welcome to the Pytorch dev podcast.
Today, I wanna talk about double backwards, the way that Pytorch implements higher order differentiation in Pytorch.
What's higher order differentiation? Well, normally, we think of differentiation as just the thing we do in order to figure out you know, how we wanna update our gradients and parameters.
And, you know, as machine learning people, we just leave it at that.
Right? Like, you know, it's just an optimization problem.
But, you know, differentiation comes to its roots in calculus.
Right? Like, it talks about the rate of change of quantities.
And if you can talk about the rate of change, of a quantity, you can talk about the rate of change of the rate of change of a quantity and so forth and so forth.
So like, you know, in high school calculus, Right? You you can have a function that model's position, differentiate it and you get velocity, differentiate it again, you get acceleration, etcetera etcetera.
So what are some use cases for higher order differentiation in deep learning? Well, there are many use cases of this actually.
It's actually a very popular feature, although it doesn't show up in like simple models.
So one good example of this is this concept called gradient penalty.
The idea behind gradient penalty is that sometimes when you are working on your model, you will have a example that causes the gradient to have a really, really, really huge step.
And maybe that's bad.
Right? Maybe you just don't want to do that.
Maybe you wanna make sure that any given input doesn't influence the state of your parameters too much.
And so the bigger the gradient, the worse the solution is.
Well, if you're just, you know, doing a good old fashioned single order differentiation on your program, then there's nothing you can do.
Right? Because you just compute the gradient and then, well, you got your gradient.
May maybe you can just clip it before you actually apply it.
but what you can do if you have a high order of differentiation is you can actually apply a penalty.
You can say, hey, so I want to reduce this loss but I don't want to reduce this loss if it will cause the gradient to blow up too much.
So I can have a like combined loss that takes into effect both the, you know, loss and question, whatever it is that I wanna train on, you know, the accuracy of my network.
But also, we'll successively penalize if, you know, the gradient gets bigger and bigger.
And I can then, you know, via the magic of automatic differentiation, find the exact quantity that will minimize my, you know, sort of, joint loss involving the true loss as well as the penalty on gradient.
And how do I do this? Well, I have to do this with higher order differentiation.
Right? I have to first differentiate my program to get the gradient and then I have to use the gradient with my regular loss and differentiate again to find out how I can minimize this combined loss.
Another example of higher order differentiation being useful is in meta learning.
So what's the concept behind meta learning? Well, meta learning as the name suggests is learning to learn.
So it's all about, you know, training a neural network to train a neural network really good.
And what does this often look like? Well, you know, normally, when you think of how you differentiate a model, you have a training loop, and what you do is you, you know, you run your model forwards, you run the model backwards, you get the gradient, you apply the gradient to the optimizer, and then you you go back to the loop and you go again.
And then there there are gonna be some hyper parameters associated with this training loop.
And typically, you just have to find those by, like, just trying a bunch of things.
You know, like, change the hyper parameter and then try again.
Well, in meta learning, what you'll do is you'll, you know, run this training loop and then this training loop itself, you will run an optimizer to optimize, you know, some hyperparameter, maybe some aspect of the model architecture.
And that that so the entire training loop is embedded inside a bigger training loop which is training, you know, the overall, you know, how well the neural network learns in this case.
And once again, you know, you have to do a normal gradient commutation inside the inner training loop, and then the outer training loop needs to, you know, do a gradient again on the inner training loop.
And one last thing.
It's commonly the case that you might need to compute a session when you are, you know, doing some mathematical applications.
That's the square matrix of second order partial derivatives.
Second order means you need higher order differentiation to actually compute this value.
So you just can't do it unless you have support for this.
Okay.
So how does double backwards actually work in PyTorch? So this is gonna be a long installation.
So I'm gonna take it in parts.
So the first part is I wanna first explain how regular AD works.
If chances are if you've, you know, seen any, like, in-depth tutorial on Hydrogen.
You already know this, but it's gonna set the stage for dull backwards.
Next, I'm just gonna introduce how exactly the double backwards user API works because it says something about the implementation.
And then finally, I'm gonna tell you how double backwards.
Okay.
So let's get started.
So how does regular automatic differentiation works? So what's the model? So the model is you have a bunch of parameters these parameters are written with requires grad true.
And then whenever you do operations that involve these parameters, you record information about what operations were done.
You know, in in the literature, this is a call, you know, writing it to the Wenger list that, you know, records all the operations.
So we record operations as we execute them.
And then when finally, we call backwards on the loss we traverse this graph in reverse order and run the operations in, you know, sort of, backwards order computing the derivative, propagating it through until we get the gradient in the end.
There's a lot of math that explains why it's goes backwards and, you know, what exactly the meanings of these operations are, but this is not sort of relevant for just understanding how double backwards work.
There's two other details about this process which are worth noting.
So one is that whenever we, like, process this graph backwards, we actually eagerly deallocate the recorded gradient info whenever we're done processing it.
And why is this the case? Well, because normally in a normal training loop, you run backwards once and then you just use the grads that are accumulated into the parameters to actually do the optimizer update.
So you don't actually need this, you know, sort of reverse graph anymore.
Right? Like, once you've used it, you're done with it and you don't need it anymore.
So we can say memory by just deallocating it as we go along.
And it's also really useful because if there are reference cycles while deallocating the ground info can break those reference cycles.
Second is that there is something very interesting that goes on when we run the backwards, which is that The backwards formulas for various functions may involve uses of the inputs in question.
Right? Like, if you multiply a times b, the gradient is grad a times b plus a times grad b.
And if a was a parameter, well technically requires grad equals true, says that you're supposed to court, grab info for this situation.
But we don't do that because, like, it's very unlikely that you're actually gonna, you know, run backwards again.
Right? You're gonna throw everything away and then run your PyTorch program again on the next batching question.
So by default, we disable the propagation grid info when we're actually executing backwards.
Okay.
So hopefully, you can see where this is going.
So when you wanna use double backwards in PyTorch, The user API for it requires you to do two things.
So one is it says, okay.
First, you have to pass this flag called retain graph.
what does retained graph do do? It says, don't get rid of the graph info as you process the backwards in question.
Why, you know, is it important to retain the graph info? Well, it's because, you know, when we do a double backwards, we might need it again in that situation.
And the second thing they tell you to do is to pass in create graphs equal true when you run backwards.
And what does that do? It says, oh, hey.
Actually, please do report gradient inflows as you compute the gradients through the backwards graph in question.
And once again, why is that useful? Well, it's because you're going to want to differentiate it through later.
And so what double backwards then says is, okay.
So you you run backwards with these two arguments.
And then at the end of doing the backwards, you get a grad.
But this grad actually has a grad info on it.
it has recorded all of the history necessary in this case, and you can now use it as part, like, for example, gradient penalty.
Right? So now that you have the grad, you can add it to your loss.
And then this entire mondo thing, you can actually just go ahead and do another backwards on it.
And this is why we call it double backwards.
Right? You call backwards once, you get some grass, you do some stuff with the grass, and you call backwards again.
And that's the double quest Sometimes I find this process a bit mind bending, and one of the things that, like, sort of, helps me retain my sanity when this happens is I imagine that actually when I run the backwards the second time, I don't actually care about the first backwards.
as in, I can reason about the second backwards without making reference to the first backwards.
Why is that the case? Well, let's imagine that instead we were doing a functional transformation on our program.
So once again, I'm using the sort of Jack's terminology, but it's really useful because it gives a good idea intuitively of how this all works.
So in my basic patterns program, I write explicitly a bunch of operations that perform the forward pass forward operations one by one by run.
Right? Like, you know, take my parameters, you know, do some convolutions on them with the inputs, etcetera, etcetera, until I get a loss.
And this is my program.
And then in PyTorch, you just have to write dot backward and then it gives you the backward.
But when we, you know, tell people about how AD actually works, we say, you can imagine this backward call expands into a second program like imagine copy pasting in the second program after your first program that goes ahead and runs all the steps but backwards and with all the operations replaced with their gradients.
And so, you know, this this composite program involves running windows and stuff forwards, running stuff backwards.
But if you look really carefully at all the operations and the question, they're just good old fashioned, you know, operators on Pytorch tensors.
The backwards functions are not anything special When you take the gradient of a multiply, it just uses multiplies and adds.
When you take the gradient of a convolution, you get, you know, convolution backward, but convolution backward is a good old fashioned function.
And more importantly, it itself has a gradient, so you can differentiate that as well.
And so whenever I have a double backwards that happens in this situation, I just imagined this, you know, big graph, right, that has forwards and backwards And then I just forget that I ever knew that, you know, this was a separate folding backwards.
I just imagined that some poor grad student in the eighties had to, like, manually derive all the backward steps themselves.
So I I've just ridden all out is this opaque program.
I know nothing about it.
and then I just apply automatic differentiation to this program again.
And well, lucky me, You know, what does AD know how to do? Well, it knows how to handle any any program that consists of a bunch of operations that, you know, affirmatively, I know how to differentiate.
That gives me my double backward program.
And actually, when I'm, like, reasoning about what graphs look like in double backwards, I like, you know, writing a simple gradient penalty example.
you know, writing out the backwards and then writing out the backwards of that, you know, forwards backwards program.
And that gives me a graph and I can use usually use that to reason about some of the weirder things that happen in this situation.
So on the one hand, we're done.
Right? Like, double backwards is just, you know, doing backwards again.
you know, what's the big deal? But actually, there's a reason why higher order differentiation is kind of mind bending to implement.
from a, you know, if you're just purely looking at it from a Wenger tape perspective.
So one of the things that is really mind bending is that when you do higher order AD, you actually need to reuse things from the graph of the first eighty.
That that's why we had to do retain graph.
We're not allowed to throw away any of the grad inflows for the original program because when I'm looking at my backwards program, Well, you know, one is eventually things go back to the loss.
And, sorry, not the loss.
eventually, things need to make use of various parameters that may have been defined by the original graph in question.
So going back to the multiply example.
Right? The gradient doesn't only make reference to the gradient of x and the gradient of y.
It also makes you reference to x and y through the, like, derivative formula says, hey, you need to know what these quantities are from the original network to actually confuse the gradient in the situation.
And if those things require a grad, then when I use them in the backwards graph, then I need to, you know, go keep going past them.
Right? Like like it's a data dependence and when I do backwards on my, like, composite program that sends me back to the original graph.
And so that's pretty important and because it it's very interesting what happens in the situation.
which is that when I go and I traverse parts of the backwards graph that were used again for the backwards in question, I have to flip it again.
I'm not explaining this very clearly I'm just gonna leave you with a very impressionistic picture of what happens.
So you have a forward graph.
Right? When you differentiate it, the graph sort of turns upside down because your backwards graph is exactly the same thing as the original graph, but going back in the reverse direction.
When you differentiate that graph again, well, you flip it back and it looks just like your good old fashioned forward graph in question.
It's actually the linear approximation because, you know, you're you're not only doing this flipping, that's the part where we're doing reverse mode a d.
but you're also taking a linear approximation.
So what happens, in other words, what happens in double backward situation is you end up having to recomute the original forward graph, but with different parameters for, you know, what the inputs are because they're coming from different places.
in your graph.
And this is one of the reasons why symbolic automatic differentiation, which might want to be done by systems that aren't tape based have such a, you know, sort of, like, it's actually really tricky to do it all correctly because there's all of this stuff going on And you can't just assume that, like, you know, when you had some program that you compiled for forwards, that's it.
That's the only thing you need to compile it for.
The the sort of transformation can reuse things, you know, unpredictably.
And one of the really nifty things about Pytorch's design for double backwards is it can support an unlimited number of double backwards operations as long as you don't ever clear the graph when you do these things.
And sometimes in other situations, when you want to, like, optimize You have a problem which is that you need to know ahead of time how many times you're going to differentiate your program because if you don't know that, then you can't actually, you know, safely get rid of variables because they might get reused at higher levels in question.
Alright.
So I won't claim that you will completely understand double backwards at this point, but hopefully I've given you the main idea, right, which is that you know, you don't have to think about double backwards as this mystical thing.
It just is running backwards again on a program that happens to have been generated partially by backwards.
But this actually causes some very intricate behavior if you actually wanna dig into it.
But at a very high level, and when you look at the implementation, that's all there is to it.
Alright, everyone.
That's everything I wanted to say for today.
Talk to you next time.
.
