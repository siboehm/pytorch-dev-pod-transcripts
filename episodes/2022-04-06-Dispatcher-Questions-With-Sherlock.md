---
layout: post
title: "Dispatcher Questions With Sherlock"
date: 2022-04-06
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Dispatcher Questions With Sherlock

Alright.
Hello, everyone.
Today, I'm here with Sherlock Wong, who is newly joined here at meta and this is an interesting new format that I wanted to do.
Basically, Sherlock is gonna ask me questions about things in Pytorch from a sort of newcomers eye.
Although, Sherlock, you're not really a noovie because you've been working on Onex runtime for quite some time before coming over here.
And I'm gonna answer them, and we'll see how this goes.
And today's today's episode also has a video component with it.
because there are some diagrams that we're gonna reference as we're going.
Alright, Sherlock.
So do you wanna go start it? Yeah.
Thank you, Howard, for inviting me here.
Yeah.
So today's we're probably gonna focus on the discussion.
component.
So and so I was reading your blog.
So I come to this impression that originally, the sculpture was designed to handle mostly just, you know, the device type and data type dispatching.
So over time, it's growing to this big magnetic and central place for many, many features.
So I can tell me a little bit about history of how we come to this stage and a little bit of the history about the dispatcher.
Sure.
So your guess about where the dispatcher used to be eliminated is right.
So in the beginning well, in the way way way we beginning, we we had we had torch.
It was for Lewis torch.
It was written entirely in sea.
And essentially, all we had was we had, like, separate copy pasted files, one for CPU flow tensor, one for CPU double tensor, one for CUDA float sensor, one for CUDA double tensor.
And then there was just some binding to the Lua program language that actually, like, figured out where you would go so you didn't have to, like, write individually which operation you wanted to do.
So this got ported to Pytorch.
And so the first version in Pytorch, there was some binding layer in Python that once again basically knew how to get to the right implementation.
And when Zachary DeVito rewrote our binding so that we had a c plus plus library intermediating between Python and the torch c libraries.
Before, it was like directly to see and was very hard to understand.
The original thing that we needed was simply, yeah, to to dispatch on the device type.
and then to dispatch on the d type.
So there was a virtual method that we used to do the device dispatch.
and there were a bunch of macros for basically letting you stamp out multiple copies full of each implementation for d types.
So since then, we've added tons and tons of more features to this dispatcher.
And I do have a there's a more recent diagram talking about dispatch keys.
And the the model we have now in c plus plus is that there's an order of various operations that we can do in the dispatcher and we want to we basically run things in order depending on whether or not they're applicable to some some computation or not.
So like in this example, Autograph is in red, and that's because Autograph is a very common layer people wanna do.
So you hit Autograph.
and then you do CPU.
So Autograph was like one of the first layers to get added afterwards.
And then all of these other ones sort of got added over time.
Did that answer the question? I guess I I didn't answer this question you had over here, which is how to, like, think about vmap.
But at least, Auto Grad, that's where it lives.
It lives here.
By the way, torch dispatch, it it's like a back end, so there's just a torch a python key over here, and that's how torch dispatch got gets handled.
in the back end section of the dispatch? Yeah.
Balances question.
So but it seems to me that, like, the order of the dispatch key is extremely important.
So as we add so many features on how do we determine the order of the inspection? You know, like, is there any press code behind you know, making the deciding the order? This is a great question.
So the order is indeed important.
And in fact, there is not a single well defined order necessarily in all cases.
For example, a functorch which is the new library for doing textile transforms on PyTorch.
It provides two levels of functionality, vMAP and autograd.
and you can actually have them ordered one way or the other and these correspond to different but both useful operations.
One of them compute per sample gradients, whereas the other one is like, you know, normal, you know, you have a batch computation inside of autograph.
So so that's some that's kind of troublesome.
And, like, some of the more recent work has been about, like, sort of getting us away from this fixed set.
But the the order that is in c plus plus and this order is kind of important because it's the one we can efficiently implement.
This order is basically sort of worked out based on, like, what the average use case in Pytosh is.
So for example, one there's a there's a a question about tracing.
versus auto grad.
So why why is the tracer before or after auto grad? Well, actually, The tracer key is this interesting legacy concept for torch grip tracing.
And we've been talking about this new thing called AOT Autograph, which knows how to trace Autograph, and that's implemented using the Python key pipeline keys after Autograph.
So indeed, you get the traced forward and backwards in the situation.
Another example is AutoCraft and AutoGrid.
Auto Cast is before AutoGrid.
Why is that the case? Well, it's because if you do a bunch of casting, You need to also know how to differentiate through a cast to lower or higher precision.
So auto cast doesn't handle that.
It just you know, inserts the new operations and then Autograph handles it.
So there's a lot of thinking about like what you want the semantics to be.
And the dis the ordering of the dispatch queues is like our best guess about what you want in this situation.
And hopefully, it is useful, but sometimes it's not.
Yeah.
This is correct.
So, like, I also heard about, like, in terms of his statute, there are two category URL.
Back end related especially, and then there's another feature keys.
And back end keys is always the end destination of this patching.
Right? That's right.
Well, almost.
Because the python key, which we treat as a back end, can in fact start executing other PyTorch code, which will go through the dispatcher again.
It's a sort of reentrant mode of execution.
But most normal back end keys don't do that.
They just actually do the compute.
So let's say, like, user want to specifically override the discussion order.
Is there any way that a user Probably can't do that.
That's also a good question.
So in the c plus plus dispatcher, there's a fixed order and that's it.
You you you're out of luck if you wanna reorder things.
So how does Fungtorch do it? How does Fungtorch let's say, do I have batch yeah.
I in in in this picture, batch is before autograph.
So that's the order you get when you use Pytorch only.
So how does Fungtorch let users reorder it? So the basic idea is that you you have you have an inner tensor and you have an outer tensor.
And each of these tensors gets its own copy of the dispatch tree.
And then what you do is you say, okay, on the outer tensor, skip bathed and go to autograph.
Cool.
You do your autogarty stuff.
And then you go to the back end key.
It's gonna be the Python key or in in Foamtorch's case, there's a special key that they've got.
for, like, going back to the front.
And this goes to the inner tensor.
It goes back to the beginning and then you, this time, hit batched.
And then finally, you get to the true back end.
So you basically like, if if it's not in the right order, you just stack as many of these as you need sort of nulling out all the things you don't care about.
And all of these, you know, layers are optional.
You don't have to do them if the functionality isn't relevant.
Yeah.
This is the same idea, similar idea as the cancer subclosing product.
So when you rep cancer, over cancer, over cancer, you automatically get multiple stacks of this dispatch key, and you can compose them in any way that you wish.
That's right.
So with tensor subclasses, you so Foam Storage's implementation is done in c plus plus.
But with tensor sub classes, you can do it entirely in Python, simply by having a tensor sub class that contains another tensor on the inside.
And we actually have an example of how to do that this way in sub class zoo.
It's in the functorch dot pie file.
Yep.
So I wanna dive a little bit into this back end Slack key.
So So in in the in the in the diagram, it sends that, okay, we do a bit wise for all the hardware keys.
but end up the one that end up being selected is the leftmost of the dispatch key.
Right? But in reality, like, a single operation can only run or a single device.
So, like, way, the multi dispatch behavior doesn't apply to the back end select.
Is that the right on standing? Sort of.
Okay.
So there are a few things going on here.
So so this multi select is for handling multiple dispatch where you, like, have a CPU tensor added to a CUDA tensor.
And this figures out, hey, you wanna go to the CUDA key, not a CPU key in that case.
Back end select is for a different situation, which is when you don't have any dispatch keys in the inputs in question.
So back end select is used primarily for factory functions, which don't take any tensers as inputs.
And because they don't take tenses as inputs, there's no tensor in pet dispatch keys to get you to the CPU or crew to factory function that you need.
Indeed, you would instead have to look at the device argument to figure out which one you want.
But we didn't write any like special logic.
We just like, oh, if your argument is a device, then I know how to extract a dispatch key from it.
Instead, we just said, well, we'll just put you in this back in select kernel.
It will go and look at the arguments, figure out what to do.
and then eventually take you to the correct kernel.
I see.
So for let's say there is a binary op.
And one input has a set of dispatched keys, say, batching and tracing But on the other input, it can have another set of dispatch behavior.
So when we pop this two inputs into the this binary off, so you're saying that both it would take a union of this dispatch key and in both every single feature that those of them have.
Right.
Yeah.
That's right.
And then the implementations of the feature, like batch to auto grad, would be responsible for knowing how to deal with a tensor input.
that quote unquote wasn't batched or wasn't autogrouted.
In in jacks, we call this lifting.
You you have to lift a tensor that doesn't have that functionality into the world of autograph or batching.
I see.
I see.
So it seems to me that, like okay.
The destination is always on this back end device selection power.
So has there been any consideration on -- Mhmm.
for example, breaking this one joint, this battery into multiple wands, especially, for example, for the back end wands.
because it's always the destination.
It doesn't seem to be mixing with other orders.
Yeah.
with the layers.
This is a great idea, and in fact, we have done that.
So Brian Hirsch recently landed APR that gives us a lot more dispatch key space.
And the way that he does it is by treating back end especially so they don't you you can sort of have something that is both AutoCast and Auto Grad and XLA, but you can't have something that's both XLA and CUDA.
So he encodes those differently.
With it's still one in sixty four, so we didn't actually separate them.
And the dispatch table is still sort of set up as one table.
But but more early now, we are treating back ends differently.
then then the layers in question.
I see.
Is there any other, like, category concept in this discussion kit for example? Like, we have this multiple autograph, this patch key.
We have this view and conjugate and and and negative view.
So it seems to me, like, the dispatch key is not, like, completely flattened.
There are a structure within it, but somehow it all just appears to be flattened.
So is there any consideration to put them in a more structural way? That's a good question.
So for the layers, I actually don't think there is more structure except in the sense that you might wanna reorder them arbitrarily, which is what Fungtorch is about.
So so my general way of thinking about every layer keys.
So the so back end keys are terminal.
Right? They don't ever call anything else.
Layer keys can call into other layer keys.
and we sort of normally go down the dispatch key chain as we get there.
So the way I think a little layer key is it's basically a rewrite of some torch API operations, some eight ten operation into some smaller eight ten operations.
Right? And you continuously, like, d sugar or the eight and offs into more and more eight and offs until you finally got in the back end, and then those are the actual operations that you need to run.
So in in that sense, there isn't really any meaningful grouping.
Right? And any any transformation from a single a ten up into several a ten ops is a valid transformation.
And we might group them up because some of the transformations do similar things are implemented in a similar way like conjugate.
and negative.
Those are very similar.
But, you know, like, you don't need to you don't need to bunch them up from the perspective of dispatch.
because they, you know, they they do wanna be ordered in this way because that tells you what are the transformations happen.
I guess there is an interesting point here to be made, which is whether or not Sometimes transformations are commutative with each other, but we we don't encode this logic in any way right now.
So another mystery that seems to me is, like, all this new feature end up landed in the dispatcher.
So is it by design or is it just part of the constraint? that will end up in the disclosures.
So, like, if you look that into all this feature that was added to this disclosures, is there any particular ones that could have lived outside the dispatcher or done in a different way, that but eventually somehow still got into the dispatcher.
That's a good question.
So let's let's be a little concrete for a moment.
So let's say for example, Let's talk about AutoCast for a moment.
So AutoCast was an interesting feature.
It wasn't even developed by folks at Facebook.
Michael Corelli over at NVIDIA had implemented AutoCast as some I don't even remember how he I think he was, like, monkey patching, the pipe torch source code and basically doing a lot of work to, like, basically automatically insert casts to lower precision you know, without having to modify your program.
And so I think we were at the Pytorch dev day and I I heard what Michael was doing and I was like, hey, you know, you don't have to do it this way.
We've got this thing called a dispatcher And in particular, something you can do in the dispatcher is you can write what's called a fallback kernel.
So this is a single polymorphic kernel that, like, will work operate on all of your operators.
So you don't you can just write one of these fallback kernels Actually, I think I think for AutoCast, it's a fall through kernel.
It just ignores the execution if you you know, if it's not an operator that knows how to do casting.
And other than that, it was just a very convenient way to insert functionality into Pytorch interpose it, you know, without having to, like, do stuff like monkey patching.
Now, today in twenty twenty two, we've been adding a lot of new features that let stuff happen in Python user land.
For example, I've got a PR, a torch function mode which I'll be landing soon, which basically lets you do this kind of interposition at the Python API level.
And this can't happen in c plus plus because once you've gotten to c plus plus, we have only the, like, narrow c plus plus set of types.
So all Python objects have gone away we've translated them into c plus plus.
But sometimes you wanna, like, do some operations in Python and that's why to function modes are kind of like dispatcher layers on except they're living at the Python level.
And then you probably could have implemented AutoCast.
in Python.
And you would only do it in c plus plus because you had a speed concern or you needed to work with the c plus plus front end or something like that.
So yeah, it's a really good question.
And so I guess historically the answer is there wasn't a good way to do things other than in the dispatcher, but we are now adding more hooks like Python dispatch and like a torch function mode, which lets you do these in user LAN.
Yeah.
Yeah.
I think that we're that's all the question I have for today.
Okay.
Thank you, Sherlock, for asking some great questions.
Talk to you next time.
Thank you.
Bye bye.
.
