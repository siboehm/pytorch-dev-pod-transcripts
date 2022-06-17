---
layout: post
title: "Torchscript"
date: 2021-06-13
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Torchscript

Hello, everyone, and welcome to the Pytronch Step podcast.
Today, I want to talk about the torch grip compiler, also known as the Jit Compiler.
It's a little bit cubistic to think that I could explain an entire compiler stack in a fifteen minute podcast but I'm gonna try it anyway.
Fortunately, there's plenty of resources for you if you want to dig in deeper than, you know, the short amount that I'm gonna be able to talk about today.
In particular, there is a really good overview document in the JIT directory, also linked in the episode notes, that is basically gonna cover everything that I talk about in this podcast and everything else in more detail.
So my goal here is to sort of just give the big picture, tell you a bit a little bit about the torch grip compiler even if you you don't necessarily know anything about compilers.
So it's gonna be a mix of here's what a compiler is.
And also, hey, you know, here are some of the interesting things that are going on in torch scriped specifically compared to other compilers in question.
So I'm gonna structure this like you would structure a normal compilers course don't tell William Bowman, I said that he structures it in the other direction, which is we'll start from parsing and then go successfully lower as we, you know, progressively lower into simpler and simpler registrations, until we get to the interpreter, which actually is responsible for running your tour strip programs.
And so each step is like a traditional step in a compilers course where, you know, first we do parsing, then we do lowering, Then we talk about optimizations, then we talk about actual code generation.
Although in the case of PyTorch, we don't actually generate x eighty six code from your transcript models.
We just interpret them.
And each of these steps will talk about, you know, some of the things that are going on and how to understand what they're doing.
from the perspective of a PyTorch developer.
So let's start by first setting the stage for what JavaScript is about.
So what is towards script? So towards script is a way of taking python programs that you've written and re representing them into a form that, you know, is not python, but is an actual honest to goodness IR that we can do optimizations on, which we can easily package up and send and run on say c plus plus services that don't lay against Python at all.
You may recall in a previous podcast I talked about torch deploy a technology for making it possible to run Python programs from multiple threads inside a single server process.
Before torch deploy, there was torch script, and torch script took a much more direct route, which was saying, hey, if you wanna run your model in a multi threaded fashion, we're just gonna get rid of pipe Python entirely.
And so in order to do that, we need to actually have some way of representing our program and have it be rungable from c plus plus without a python interpreter at all in the loop.
Oh, and by the way, we also wanna do some optimizations like fusion to make our programs run faster in this situation.
So Torescript, in other words, is a graph mode for Pytorch.
Pytorch is all about eager execution, but Torescript actually lets you tick your hydrogen program and put it into a machine processable graph representation that we can do transformations on and that we can actually execute in this way.
But there's a step further with the Torsco compiler, which is that we want to actually be able to capture the control flow and other sort of features of people's programs that otherwise you couldn't get from, say, just tracing your program.
So in the very first version of Tor script, we implemented getting Pytrish programs simply by just running your, you know, eager penetration mode program and seeing what operators were called, and those were the operators that we've actually put in your trace.
So Torescript wants to be able to handle your beam search or your while loop or your, you know, if conditional.
It wants to handle all of those things.
And so it basically wants to capture a kind of, you know, high fidelity representation to your program even if you know, on a single eager remote execution, you might go down one path or you might go down different different path.
So it wants to capture something that can describe all of the possible traces of your program in a situation.
So with that in mind, what this means is that when you talk about the transcript compiler, you have to talk about an actual parser.
That is to say, you know, we can't, you know, do the easy trick, easy way out of just tracing your code and getting out of representation of all the things that get got run-in runtime.
Because as I said, there might be an if condition, there might be a while loop, and there's no way to trace all the possible different versions of it.
unless you're in a language that supports abstract interpretation, which Python is done.
Okay.
So so what does this parser look like.
So we've got our Python code and we basically need a Python parser.
And so in fact, there's two parsers that Torstrom support.
There's one parser that's written in Python and it's based entirely off of the stock Python AST module that lets you, you know, take some Python code and blurts out an AST.
We also have a reimplementation of this in entirely in c plus plus, so it's a lecture that's a thing that takes in a string and reduces into a bunch of tokens so that the parsing stage which organizes this into a parse tree can do it more efficiently.
We have a parser and we have a parser that knows how to parse Python implemented in c plus plus And remember, that's because JavaScript is all about being able to run Pytosh programs in context where Python is not allowed.
As a side note, this actually is very important code because we don't actually serialize some sort of, you know, random byte code format when we wanna save towards script models to disc and, you know, remember, this is one of the things that, like, JavaScript is designed to do.
It's, you know, take your model put it into some format so you can load it up into the model server somewhere else.
We actually save honest to goodness Python code as our serialization format because it's easy to debug It's easy to modify if you need to.
You don't need special tools to deal with it.
But, you know, that's only because we're on server and it's not a big deal to parse hyphen code when you're loading up your model.
On mobile, where binary size is at a premium, see my mobile podcast episode we don't wanna pay that.
And so there's actually a different version of the serialization format that's used by mobile And that's actually a, like, good old fashioned bytecode format that's easy to parse and so you don't need, you know, something that understands Python syntax to parse it.
Okay.
So you you've done the parsing stage of your program.
Right? And so given the Python program, Now you have this AST that looks a lot like the surface syntax, but it's, you know, in tree shape, it's easy to look at.
and it's got all of the language features from Python that, you know, you actually support.
So, like, if you got a while loop with a break, you know, you're gonna have a AST that has okay.
Here's the while loop, and then inside it, there's a statement, and that's the break statement.
And so the next thing you need to do in any honest or goodness.
Compiler class is you want to take this, you know, sort of direct reflection of the surface syntax as an AST and lower it de sugar it into a simpler representation that's easier to do processing on.
This is just like, you know, the very standard thing you do in compilers because People want tons and tons of features in their surface language.
Right? Like, the more features, the better.
Like, invent a new syntax, you know, do all sorts of fancy things.
And as a compiler writer, like, this is a big problem for us because we need a write code that can work no matter what features you use.
And so the easiest way to make our life easy is to, you know, take all of this the surface syntax that, you know, all our users want, and then condense it down into a smaller set of syntax that, you know, we only we only have to worry about when we're running our passes.
So there there's this transformation.
There's a bunch of optimization passes because sometimes we have to do non trivial analyses to figure out how to, like, re rejigger things into the simple format.
But eventually, you get to what we traditionally called transcripts i r.
So what is transcripts i r? So if you know what LLVM, IR looks like JavaScript IR has a lot of similarities to it.
It's SSA.
That means that for any given variable you define, there is a single definition site for it.
So you don't have to worry about, you know, you you're, like, you're an optimizing an optimization pass you're like, okay, who define this variable? You don't have to worry that there's multiple possibilities like one in this if branch and one in this else branch.
SSA means that, oh, yeah.
There's only ever going to be one of these things.
Another thing about the Tortue IR format is It does understand conditionals.
These are actually added after the fact because remember, tracing, you don't have any conditionals.
They all go away.
And the way they're modeled is that instead of a good old fashioned, you know, CFG style setup, where you have a bunch of blocks and they have labels, and then you have fine nodes for when blocks enter in from multiple possible entry points.
Instead, what we just do is we we we it's more of a structured workflow flow control style where, like, when you have an if statement, There's two subblocks associated with it that represent the conversation that gets run-in the first case and the second case.
And they you're responsible for passing in the inputs.
And then when you exit, you have to say all the variables you want to return.
And then the if statement itself does return values and it returns all of the sort of values that get carried out of the loop.
So unlike LVM, SSA, we don't have fine notes.
Instead, those are sort of done implicitly.
via disease.
What what they're what they're known in the literature as basic block procedures in this situation? two more important things to say about the element i the the transcript i r.
So one is that although we simplify the aspects of python programming language, so we'd have less features.
We still have a really big instruction set.
Every there's a bunch of, you know, like, when you have an i r like this, there's there are gonna be operations, primitive operations, or prim ops for short, which don't have an implementation inside of the IR itself.
Instead, the compiler stack defines what these operations should be.
And every operator that's defined in Pytorch.
We call the native functions dot yamal podcast, that that's the list of all the operators.
Every operator is a valid instruction inside oftorch script i r.
this is kind of a pain in the neck for compiler writers who don't really want to, you know, deal with, like, over a thousand operators.
And hopefully, in a future podcast episode, I can talk with Zachary DeVito about some recent work he's been doing about mentor, which is reducing the set of operations in Pytorch.
But okay.
So we have this really big primitive operator set, but it's in SSA form.
We've got control flow in a structured manner.
simplified.
So there's only a few control flow operations.
And one last thing is that this IR supports mutation and aliasing.
What do I mean by that? So when you write Pytrich programs, you can take out views on tensors.
Right? Like you can say, tensor open bracket, zero close bracket.
And then it'll give you the 0th row of your tensor.
And if you mutate that, like you say, x dot add underscore law, then the base will get updated as well.
And towards script can handle programs that do mutation.
It can handle programs that do views and we don't have a functionalization pass that removes all of these things.
So the IR needs to also know about the concept that Some operations might have side effects, no, you cannot move operations around Blyeli because, you know, if you move a use of a tensor before, you mutate it, that's you're gonna see the old version, not the new version, that's gonna be semantics changing.
So really, like, what towards script i r is, And maybe this is not the best point in the design space, but it makes a lot of sense if your goal is to like, package up as many Python torch grip models as possible into this representation is, you know, we support all the operators in PyTorch and we support a bunch of control flow, we support mutation interviews, and we but otherwise, it's an SSA format.
So it's still possible to do optimizations on this.
So once you're in an IR, the next thing you wanna do are optimizations.
And we do all the sort of basic optimizations like people optimization, that sort of thing.
But there's two really interesting optimizations that are very specific to PyTorch.
So one is specialization.
So what do I mean by specialization? Well, when you write python code in PyTorch, typically, you don't give it very detailed types.
Right? Like, for example, you have a bunch of inputs and they're just tensors.
And you don't really know anything else about what they are.
Actually, in reality, they're probably all floating point tensers of, you know, dimension three, but you don't we don't know that when we're parsing the transcript i r.
And so there's this concept called the profiling executor.
We wanna sec.
So this is a bad thing for us because if you want to optimize your code, if you want to generate kernels, the more you know about what the d types are, the sizes are, the dimensions are, the more you know about these things, the easier it is to generate good code.
Like, let's say you're doing like fusion and you wanna fuse a bunch of point wise operators together.
Well, you can't actually do it unless you know, for example, what the dimensions of things are because if the dimensions don't match, you might actually need to do some broadcasting in this situation.
So what the profiling executor does is it runs through your code on some inputs and it says, oh, here are what the types of everything are.
And then it introduces this information into the Wordscript IR, and it does so in an interesting way.
So it's not the way you would think, which is, like, take your IR and then generate a specialized version of it.
Instead, we take the i r as is and we insert a bunch of what we call guard statements.
What these guard statements say is, if it is the case that something has this type, then do this otherwise bail out and do something else.
And so inside the segment of the code where the guard is okay, we actually now are able to optimize under the assumption that, you know, it's floating point and has these sizes.
And at the same time, we haven't changed the semantics of the program because even if you feed it something that it wasn't expecting, you'll still get a valid result.
You'll hit the bailout path in that situation.
Another interesting optimization we do is derivative splitting.
And this is because PyTorch programs often are differentiated because you're doing gradient descent.
Now, unfortunately, Tourscript can't make use of the standard derivative definitions that are defined in derivatives dot yamal because, you know, those are basically just they're just c plus plus, and it's glorified c plus plus in there.
And torch grip is, you know, this i r, it needs its own i r definition.
So, unfortunately, we weren't able to, like, put the derivatives in a form that could be used by both Torprescript and the traditional old EGR mode.
So there's a set of extra definitions for doing symbolic automatic differentiation in Torprescript.
But these definitions are not complete because as I said, there's a lot of operators in Pytorch and it, you know, it's just hard to actually keep coverage with that.
So for the things that torch grip knows how to symbolically differentiate, derivative swing bunches them all together so that it can, you know, go ahead and generate derivatives in those case.
For everything else that it doesn't know about, it keeps those separate so that we can run the good old fashioned autograph system.
Yes.
So we're we're compiling your code, but, you know, we don't necessarily compile everything away.
And in particular, if you're going to run a d code, in towards script, we still use the egress mode autograd executor in this situation.
And so those things that don't support symbolic differentiation they'll just go through the normal autograd mechanism.
And there's a very complicated way of making sure symbolic AD and eager AD work together harmoniously in the situation.
And we should honestly write a research paper about this.
that we've been lazy and haven't gone ahead and done it.
Okay.
So you've optimized your program.
Right? So we've gone from parsing, to lowering to IR, and then we've optimized the IR.
What's the last thing? Well, program is useless unless we actually run it.
So we need to be able to run our programs.
And the way this works is, as I said, we don't actually co generate x eighty six code from your transcript programs.
Although, maybe this would be a good idea, and Some people have looked into it.
What we do instead is we just have an interpreter.
So we take our i r and we compile it into a byte code format.
It's a very simple bytecode format.
All it does is it just does some register allocation.
And the register allocation is really dumb because we we don't really we're not really storing things in, like, hardware registers.
We're just sort of, like, using the registers as an easy way to keep track of what intermediates are hanging around.
But the thing that is important about the registers is that we use them.
We need to know when tensors die.
because we need to deallocate them promptly.
And so that's something that happens during the final, like, compilation of TarshuIP IR into what's called code blocks.
So we do that.
And then to actually execute your torch good program, we do something that should be very familiar to you if you've ever started the JVM.
which is we just have a good old fashioned stack machine.
So what do I mean by a stack machine? So a stack machine works is if you wanna call a function, you push all the arguments you want to call the function with onto some stack.
Right? And you call the function, and that function is responsible for popping off all those arguments and then pushing on the return values to the stack.
Stack based machines are very nice because they give you a uniform calling convention that doesn't that works no matter how many arguments and returns you have.
Like, if you wanted to actually do it some other way, then you would have trouble, like, you know, finding memory to put all your arguments or returns depending on what the situation was.
Because remember, the interpreter doesn't know anything about what operator is gonna execute.
It's, you know, running in a loop and going over each instruction and being like, okay, now I gotta do this one.
Now I gotta now I gotta do that one.
And it doesn't know ahead of time, oh, this is a function that only takes two arguments.
And what are these arguments that we're passing in on the stack? Well, these are i values, which I've talked about in previous podcast episodes.
Right? a box representation of either either a tensor or maybe an integer or some other primitive formats.
that just let us work polymorphically over them in the interpreter.
And that's a whirlwind tour of the short script compiler.
I promised fifteen minutes.
You got twenty minutes, but that's everything I wanted to say.
And they said, check out the overview document in the Jit folder.
mean, tons and tons of details way more information than I talked about in this podcast.
That's everything I wanted to say today.
Talk to you next time.
.
