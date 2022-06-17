---
layout: post
title: "Functional Modules"
date: 2021-07-05
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Functional Modules

Hello, everyone, and welcome to the Pyturbine podcast.
Today, I want to talk about functional modules, a way of taking n n modules and turning them into purely functional, stateless versions that you can pass parameters into explicitly.
Before I start with this podcast, I had to explain why this is something that we've been thinking about recently.
So one of the projects that is going on in PyTorch is Funktorch.
Fungtorch is a sort of replication of Jack's functional transforms, but on top of Pytorch.
And one of the problems that, you know, that is challenging for JAKs is the way they have set up these functional transforms like grad and v map require you to explicitly specify what arguments you want to vectorize over or, you know, differentiate over.
And this makes it challenging to do a n n module style API, like what Pytrophage has.
I have a previous podcast about how n n modules are designed.
The short version of it is that, you know, why do n n modules exist? they exist because we want an easy way of keeping track of all the parameters for various modules in question.
And so rather than forcing people to, like, remember, what all the parameters are, you can just put them as properties in the module and then the modules will collect them all together and then you can pass them to say the optimizer you wanna do the steps.
This is really, really convenient and, you know, n n modules are a very enduring part of Pytrux's front end API design.
So what causes the problems with the functional API in question? Well, to answer this question, let's look at the sort of very most basic operation that you can do on a PyTorch program, namely compute its gradient with respect to the parameters.
Now if you think about how this is done in PyTorch normally, what you do is, you know, you have your modules, you get your input from your batch, you feed it into the modules, out pops out some final loss, and then you do dot backward on it.
Right? It's a very imperative API.
The dot backward triggers the automatic differentiation, and then all of the parameters get a grad field populated, and that's what the optimizer will read out for when you actually want to, you know, do the step update.
So there's no need to know anything about the parameters in question ahead of time or no need to actually, you know, collect up a list of all the parameters Everything will just get put where you need them to be directly on the object itself.
And so when you want to do optimizer updates, all you need to do is iterate over the list of parameters And of course, you know, what does an m module do, it lets you easily get a list of all the parameters.
Now, let's flip this over and think about what it would look like do have a version of Grad, which is an actually functional API because this is what JAKKS provides.
We also have a functional version of Grad and sometimes it's very convenient because you don't want to actually be mutating your tensers.
You just, you know, wanna get the sort of mathematical conception of a gradient.
Right? Take a function and then compute the function that gives the gradient for you.
When you're doing sort of higher order business, This is often the easiest way to conceptualize your program.
So in this setting, instead what you have is you have a function and you say, okay, I want to differentiate the output of this function with respect to some of the inputs of the function.
And now, the implicitness of n n modules is a downside because well, you know, your function normally has takes in a bunch of arguments and If you have a function that takes in everything as arguments explicitly, you can just say, okay, I want to differentiate the first argument and the second argument and the third arguments, which would just happen to be the parameters in most cases.
But within an end module, these arguments aren't arguments at all.
They are living implicitly inside of your n n module objects.
And unless you have a past that knows how to look into the n n modules and say, hey, Actually, there's also live inputs input arguments in this module object you pass into me.
there's no way that it actually will know about these things.
And so it will look to the sort of, you know, function as if these are just tensors that you're accessing, you know, sort of from out of scope.
There are, like, free variables from your function And, you know, normally, you don't differentiate with respect to free variables, except like, you know, the whole point of training your model is to, you know, do differentiation with respect to the parameters.
So actually, if you use torch dot auto grad dot grad, You can do this and the correct thing will happen.
And there's a trick that the autograd engine does in order to make this all work out.
which is that when you do a dot grad, you have to specify explicitly what arguments you want to differentiate with respect to and it doesn't matter if you actually pass them to the function or not because you don't pass in a function, you just pass in the output.
And then the autogard engine knows, like, for every input you passed in, look for, you know, the uses of it in the in the history.
And that's how things get implemented.
So there's no there's no higher unordered function per se and said we're just sort of relying on, you know, very detailed knowledge of the object identity to, like, work out what the function it is that you wanted to differentiate was in that situation.
And this trick works okay for grad and it doesn't work so great for say Jacobian.
So if you like try to do this for Jacobian, it just doesn't actually work.
You can't compute Jacobians on functions that involve n n modules.
There's also other examples of this being a problem.
So another example is when you want to ensemble model.
models.
So what is ensembling? So ensembling is the idea that more heads is better than one.
So if you had one network that, you know, was computing the answer to your problem.
Well, it might improve the performance if you have multiple copies of this network.
all with different parameters and you run them all on the input and then you sort of decide based on some voting mechanism which one you like best.
And sometimes this actually is helpful.
And there's some theorems that talk about, like, you know, idealized situations like this where they show, yes, in fact, doing an ensemble is strictly an improvement over each of the models individually.
So when you want ensemble like this, you would ideally want to run the computation vectorized if all of the modules in question were exactly the same.
Right? Because each of them is doing the same thing and you just really wanna vectorize over the parameters.
So you like you have this parameter, but it's not just a single parameter, it's a stack of parameters one per each of your modules and that's what you wanna vectorize over.
So there's another functional transformation that lets you do this.
It's a v map, but to v map a function, you have to pass in what arguments you want a v map over.
And once again, if these parameters are actually parameters in your n n module, there's no way to pass them in because your n n module is just directly accessing the parameters on that module.
And, you know, your v map has no way of sort of interposing in on it.
Because the way most of these transformations work, the way to, like, a grad transformation works and the v ray transformation works right, is that when you say you want to differentiate with respect to or vectorize with respect to some argument.
We take those arguments and then we wrap them up in some sort of special object.
like a batched tensor or a gradient tensor that says, hey, we want to do some extra work when you do operations on this.
And well, if those things are completely in the middle of nowhere on top of a module, there's no way to actually update them.
So how do functional modules work fix this problem? Well, a functional module is a is a proposal that says, okay.
Given this n n module, what I wanna do is I wanna split it.
And the way I wanna split it is I wanna first take out the parameters.
Right? Because one of the most important things the module does is give you, you know, a way to track all the parameters.
And then I want to somehow and I'll give a example of how you could implement this.
Somehow, have a version of the forward code for each of these modules.
But instead of accessing the parameters that were stored on the modules themselves and said get the parameter values from an extra argument that is passed in explicitly to the modules in question.
So you can see that if you have a way of, you know, taking a regular NN module, and turning in it into this functional version, that also solves your problem of v mapping or grouting over it because, well, the parameters are now explicit arguments.
So you can just, you know, v map over them, or grab over them, and you'll get the thing you actually want to do.
So how exactly could you do this? So, Alvin, you know, has this very simple way to do it.
Right? Which is if you want to, you know, run a module like this, you need some sort of dummy module, you get in all your parameters, you sort of edit the module to replace the parameter settings, with the explicitly passing parameters and then you just run forward.
And, you know, if you need this operation to be item potent, you you should reset the state of the module to whatever it was before.
when you're done.
So that's a very cheap and cheerful way to implement modules in this way.
And of course, you know, it might also be useful given one of these functional modules and then a, you know, list of its parameters.
It might be useful to reconstitute it back into an original NN module if you don't need this functional version in this case.
So this is a possibility.
We're not super keen on it.
One of the reasons why it's a little fuzzy to work with is it sort of it it gets rid of this notion that n n modules are objects with a, you know, sort of persistent identity.
Right? because, you know, n n modules are built out of, you know, good old fashioned Python object and oriented programming.
And in, you know, object oriented programming, when you have a object, you know, that object has some distinct identity and it's not fungible with another object that just happens to have all the same properties but, you know, is a different i identified object.
Right? Like, if you mutate one of them, you don't expect the other one to get mutated in this case.
But with a functionalization API, You're expecting to be able to, like, take these modules and then, like, decompose them into their parts or recombine them back into, you know, an end module.
and you're expected to sort of not necessarily care that the new n n module you got back is not the same thing as the one you had before.
And that is a little bit different from how the existing APIs and PyTorch work.
There's also other ways you could go about dealing with this problem.
Right? So another idea, which is a sort of API idea, is imagine that you are writing one of these functions.
Right? And instead of directly instead of directly calling into the module via some, you know, sort of, global variable.
Instead, you might be required to pass in the module as an argument into the function question.
And so the module, right, has a bunch of code, but it also is a glorified container that contains a bunch of tensors.
And so you ought to be able to say, hey, I want to v map over one of the parameters in the module in question, or I wanna grad over one of the parameters.
So so, like, sort of, instead of just having to, like, get deal with a tensor or list of tensors, module of tensors, also fair game.
Of course, once again, you still have to make sure that the if you're doing any wrapping or anything like that, you actually make use of the wrapped version of the tensor in in the internals of your function.
And this is why sort of like the the trick that we do in Pytruk for grad, which is we say, actually, the sort of association of a tensor input as actually being an input is independent of the uses in question.
Right? There's some independent weak map that keeps track of the things that are going on, that might actually be a better way of implementing like this extra behavior rather than wrapping objects in this way.
Because then, you can make sure that all uses of the object no matter if it happens to be stashed somewhere else will be run with that metadata question.
So it's very different than how Jackson, how PyTorch actually implement.
a lot of other things right now, which is like, you know, create a tensor, which wraps the other tensor in question.
One downside to this week map approach is it puts a lot of stress on how well your language supports weak references because, like, if you just used a normal map and, you know, when you whenever you, like, did new operations, you kept adding things into this map, you would obviously leak memory in the situation because you'd never deallocate and everything.
So you need to make sure that, you know, when tenses go out of scope inside your program, they also get removed from the week map.
maybe some sort of hybrid approach where, you know, inputs are done via the weak map, but intermediate results are done via actual wrapping.
Maybe that is an easy way to make sure that the memory management works out okay in this case.
As a parting note, I want to mention how the JAKKS ecosystem does with this problem.
So JAKs can't do n n modules that the same way Pytronch does, and so they have a library called Flax which, you know, gives a module like abstraction.
And sort of the key idea for their work is they just wanna completely avoid the Python object oriented's insanity.
So they're just sort of translating, you know, the code you write, which looks kind of object oriented, but is done via data classes.
under the hood, into usual, good old fashioned, pure function calls that Jax knows how to transform in an easy way.
And so, Flex actually has its own version of v map, which directly takes the module as an argument in this situation.
Okay.
So that's what's going on with functional modules in Pytorch If you have any thoughts, this is very much something that is in progress.
Richard and Horace have been working on it.
So if you have any comments please let us know on the issue that I will post in the podcast notes.
That's everything I had to say for today.
Talk to you next time.
.
