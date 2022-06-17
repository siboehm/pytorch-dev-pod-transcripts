---
layout: post
title: "Gradcheck"
date: 2021-07-23
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Gradcheck

Hello, everyone, and welcome to the Pytruch Sturm podcast.
Today, I want to talk about GRADCheck, a mechanism for automatically testing the correctness of derivatives written for functions.
Where to start on this podcast? Well, I will tell you all about calculus and derivatives and finite differences.
But before I get there, I want to talk a little bit about testing.
When the word testing comes to mind, there's a bunch of different possibilities for what you might mean in a situation like this.
Perhaps the very first set of tests, a enterprising programmer writes, or to to put it more precisely the first set of automated tests because who hasn't ever, you know, written in some code and then just run it directly in the rep pool to see if it work or not.
The first automated test one writes, tends to be of the form, write a single test case with some input and then, you know, test that.
The output is what you expected to be.
And for many types of programs, this works pretty well and, you know, if you are a proponent of, say, test driven development, the model is that you're not supposed to write any code before you start writing your test cases and you go one by one by one until you get there.
Of course, Writing tests in this way manually can be a bit irritating and it's especially bad if you're working on a numeric library like Pytorch where typically the input is going to be some random number and the output is also going to be some random number.
And it is very non enlightening.
If you write your tests as you know, if I feed in a bunch of these floating point numbers, then I get these other floating point numbers.
Right? It's just difficult to maintain and it doesn't even work that well if for example your precision is slightly off or you make a change that changes some of the Epsilon's of it all.
And now you have to go manually update all of your tests.
Now, of course, in a previous podcast, I talked about expect testing, where the idea behind expect testing is that you can automatically update your test cases when things change.
But in this particular podcast, we're gonna look at different form of testing, namely property based testing.
What is property based testing? Well, property based testing is based on the idea that instead of specifying individual input output pairs you say, hey, here is a property that I expect to hold for all inputs or maybe conditional on some properties and the input being true.
I expect to hold for all inputs and so I will simply randomly generate inputs and then see if this property is upheld by the function in question.
And when we tie when, like, in, you know, say, undergrad, they teach you about property based testing.
The like canonical example is reversing a list.
Right? So what are some properties you can test for reversing a list? Well, if I have a list and I reverse it and I reverse it again, that should give me back the original list.
Right? So reversing twice as item coded.
Of course, this is very boring and a lot of people come out of this and they think, you know, okay, well, property based testing isn't that interesting, but I want to tell you today that GRADCheck is an example of property based testing and it is really, really effective at testing programs in PyTorch.
Okay.
So how does GRADCheck work? Well, the the problem GRADCheck is trying to solve is that in PyTorch, we are a library that implements automatic differentiation.
And the problem with automatic differentiation is we basically have a bunch of mathematical functions and we never write what their derivatives are.
And in math, the derivative of function is a well defined concept.
There is one correct answer, modulo sub gradients and stuff like that.
And of course, there's always a possibility that when you write how to translate a primitive into its derivative, you wrote the translation down wrong.
And so this kind of error is what the property based testing of GRADCheck is trying to figure out.
So for a moment, let's remember what the definition of a derivative is.
There are a number of ways to formulate this.
For example, in your calculus class, you may remember some formula involving limits and f of x plus d x minus f of x divided by d x, something like that.
But another way that I like to think about thumb derivatives is, you know, suppose I have some function and, you know, it might be very wiggly, it might have very strange behavior, And at any given point, if I zoom in enough, the function starts looking more and more like a straight line.
Right? Like, I keep zooming in, zooming in, until, you know, all the wiggles go away.
Right? I'm just looking at a single segment and well, it looks like a line.
And so the derivative, what the derivative tells us is what the linear approximation of a function is at any given point on the function.
Right? So if I'm asking derivative at, you know, some point, then, like, that's gonna give me, hey, you know, like, it's flat or it's curved upwards or it's curd downwards.
Those are the various different derivatives you can have.
So it stands to reason that you can count calate a derivative simply by zooming in sufficiently on the function, looking at two points, and then turning those points into a line.
And that's exactly what the method of finite differences does.
It says, you can numerically compute a derivative by simply just taking two points on the line, the the function that are very close to each other.
dividing by the distance they are from each other, and that'll give you, you know, the slope of the line at that point.
And as as I said, that's a very good linear approximation, especially the smaller and smaller you make the delta.
Of course, Pie torch doesn't compute your derivatives this way.
It would be handously slow to do so and also not very accurate.
Instead, you know, when we write derivative formulas, what we're doing is what we're writing down what's called the analytic derivative, which is, you know, like, in math in calculus.
Right? You had a bunch of rules for, you know, giving a bunch of functions, how to convert them into derivatives.
And the analytic derivatives are just simply directly writing down those rules for automatic differentiation system.
So one of the themes in property based testing is that if you have some way of implementing a function in two ways or if you have some way of representing a property in two different ways.
If you have two new ways of doing the same thing, then a very easy to set up property in this situation is just to say, hey, when I do method a and when I do method b, they need to give the same result.
And if they do, well that's good for me.
this sort of like sort of comparison against the reference implementation is very, very convenient to test because you don't have to know anything about the outputs you just need two implementations and you can compare if they work together or not.
And so in the situation of differentiation and what GRATCheck does is GRATCheck says, hey, I have two ways of testing what a derivative is.
I can do the numerical method where I just you know, take finite differences and see what it looks like just by looking at the points.
Or I can take the analytics solution, the one that I'm trying to test the system under test.
and just directly compute it based on the symbolic formula in that case.
And all I need to do is compare these two formulas on a bunch of random inputs, and if they always agree with each other up to some tolerance, then I know that I've implemented my derivative correctly.
Of course, there's a complication and it's easiest to explain this complication in two steps.
First, When I described this function to you, you are probably imagining a squiggly line in two d space and you know the derivative was just some straight line that was tangent to the line at this point.
But first, if we think about neural networks, neural networks aren't, you know, there aren't only two parameters in a neural network that would be very impressive neural network called linear regression.
Instead, there are many many dimensions for all the parameters and they all, you know, do a lot of computation up until one point.
And so a more accurate way to think about a neural network is that you have some sort of surface, some like high dimensional surface, but, you know, as easiest to visualize this in three d space.
And what you're trying to do is you're trying to find a gradient which represents the, you know, orientation of a hyperplane, a the plane which is tangent to the surface at some point.
But that's just complication one.
Complication two is that when you look at the individual functions that are used inside a neural network, And these are the ones that we actually wanna do grad checks on.
Remember, because they're the ones who we're writing derivatives for.
It's not just a, you know, surface because there might be some sort of function which takes in some high dimensional space and produces another high dimensional space.
So really to model what this transformation looks like in a linear way at a very at a neighborhood, you need a thing called the Jacobian matrix.
It's kind of hard to describe what a Jacobian matrix does, but one of the explanations that I've read on math overflow that I quite like is imagine you you have your vector space and you're looking at one point in the vector space.
When you perform your operation on this, you map this point in the vector space to another point in the destination vector space.
And furthermore, all of the points in the neighborhood also get mapped at the same time when you do this.
And you're looking for a single matrix that describes how these points distort move around, etcetera.
And it's a matrix because, you know, we're talking about linear approximations of functions.
Okay.
So where where am I going with all of this? Well, it turns out that even when you have a n dimensional input and an n dimensional output, you can compute the Jacobian.
And the way you compute the Jacobian is you can do it both analytically and numerically.
doing it numerically, you simply just, you know, take all of your inputs, you change one of them to be perturbed slightly.
and then you keep doing this for every single input until you eventually get to the end.
Right? And so every time you perturb a different input slightly, you're getting another column of the Jacobian.
I actually always forget whether or not it's Rosa columns, but I did look this up for the podcast.
Similarly, when we do the symbolic derivative using our backwards formulas, All we need to do is for every output because remember this is a possibly an dimensional output, try saying, hey, what's the derivative what's the gradient that affects this particular output or the next one or the next one? and this one will help us reconstruct the Jacobian row by row.
And so now, well, suppose you didn't get all of that.
Right? At the end of the day, we're setting up this property based test.
And what are we doing? Well, we're just, you know, taking our two implementations, which know how to compute the Jacobian, and then just checking at the Jacobian actually equals each other.
Or at least, that's what we used to do.
So Alvin Des Mason and Jeffrey Wong came up with a pretty interesting technique for making this faster.
Because what's basically happening is repeatedly running your finite difference slash your backwards derivative on every single input slash output until you've read out every row slash column of the Jacobian.
And then you're just doing the check on the Jacobian.
So this is like very precise.
Right? Like the Jacobian is fully specified by once you read out each thing because it's linear.
Right? And the the magic of linear functions is they can be fully characterized as a matrix of the appropriate dimensionality.
But remember, this is property based testing.
Right? We're not even getting full verification that anything is right.
We're testing that the gradients line up on various points in the function space.
And in fact, it's sort of the least of our worries whether or not any particular approximation of linear approximation is correct.
Like, we don't really need to check it in all that detail.
Problemistically, we will figure it out in the end.
So this leads to the idea behind Fast Gradcheck.
The idea behind Fast Gradcheck is, hey, we have this matrix this implicit Jacobian matrix.
And previously, we were, you know, painstakingly reconstructing each of the row slash columns because that's what our things gave us.
But in fact, we don't need to reconstruct everything.
Instead, all we need to do is compute some sort of linear combination of this with some randomly sampled vector.
And well, as long as these vectors are similar, then we know that the matrices are very likely to be similar.
The reason why this works involves a bit of math and I encourage you to look at the quoted resources which talk about this in more detail but essentially what's going on is we're computing either a JVP or a VJP depending on whether or not we're doing backwards formula, that's the VJP or we're doing finite differences, that's the JVP.
And what you've got in this case is you've got the Jacobian multiplied with a vector but on one side or the other depending on the case you are.
So you just multiply it by a different vector on the other side and make sure it's consistent in both cases, then you end up with a VJU and that is going to be a very small quantity and very easy to compare.
Another analogy for this situation, which might be useful if you remember this from your probabilistic programming classes is Freeval's algorithm.
So that's given two matrices and you wanna multiply them together.
You've got some result.
and you wanna see if this result is actually the correct one.
So the naive way to do this is to actually just go ahead and do the matrix multiply.
between a and b and then compare the elements point wise, giving you c.
But what you can do instead is you can just multiply the matrices by some vector and then by the properties of associativity, you can multiply b by the vector first and then multiply that vector by a and that gives you a simple vector, whereas c multiplied with the vector direct And then you don't have to actually do a matrix multiply.
You just are doing easier to do operations that are simply matrix vector multiplications.
Same idea.
It's probabilistic, but it, like, runs way faster than the regular algorithm.
So that's the main idea behind BradCheck.
If you are very interested in automatic differentiation, I highly recommend learning what Jacobians are, what JVPs and VJPs are.
Unfortunately, a podcast is not a very good vehicle for mathematical understanding.
So if you didn't really understand all that, that's okay.
You're gonna have to spend some time with the textbook.
It's just the name of the game.
But a higher meta principle here is that property based testing is pretty cool yes, it can be hard to do correctly sometimes.
Right? Like, you need to make sure you, you know, run your random samples deterministic so that you always get the same result in CI and you also need to make sure you design your properties and your RAN numbers really well because if you don't, you're just gonna get nonsense.
But RadCheck is a really example of a really elegant way of using math.
We're we're basically like taking advantage of the fact that, you know, there's like this ad joint thing going on and that's some that relates to two different ways to do derivatives and then using that to basically test all of our functions.
So We basically don't do tests for gradients by hand.
We just rely on grad check to tell us if we got it right or not.
Okay, that's everything I wanted to say for today.
Talk to you next time.
.
