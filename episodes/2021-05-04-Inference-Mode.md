---
layout: post
title: "Inference Mode"
date: 2021-05-04
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Inference Mode

Hi.
My name is Edward, and welcome to today's edition of the Pytorch dev podcast.
Today, I wanna talk about a new feature that recently landed in head in master Pytorch called Inference Mode.
That was spearheaded by Aileen Zhang, but also, you know, had a lot of contributions from the rest of the folks.
in composability.
What is inference mode? Well, inference mode is a thing that you can do when you're writing some titles code and you want to you you are guaranteed that you're only gonna run inference on it.
And inference mode basically makes it could run faster in the situation.
It's fast enough to, like, get something like five to ten percent wins.
when we have used it inside production at Facebook.
And today, I just wanna talk a little bit about where this feature comes from, why it's necessary, and a little bit about how we implemented it.
Okay.
So first off, why does inference mode exist in the first place? And, you know, you might be thinking, hey, Edward, you know, if I just have some code in Pytorch and I don't, you know, require grad on any of my inputs, so there's no parameters.
I'm not training, I don't call backwards on it.
Shouldn't in this code, you know, just be as good as, you know, running some plain old tensor operations.
without, you know, having any support for autograd.
Like, that seems like it should be just as fast.
And, you know, if I'm a little worried about accidentally setting some requires a radical true, while there's this no grad mode, this no grad context manager, which I can already use in PyTorch to just say, hey, whatever the requires grad fields on my tenses are, ignore that and just don't require gradients.
So why is there an opportunity to make things go faster? And so it turns out that there are two things that we do in Pytorch to support automatic differentiation that can't be turned off.
They must be done because it may be possible at some point in the future that you will attempt to use these tensers for AD.
And if we don't do these things ahead of time, we're just screwed.
Whether or not this is the right trade off or not, this is historically where Patrick has been, where you know, you can always write your code and then try to use it with autograd later and this will work out.
And so inference mode changes some of these assumptions.
It says, hey, no, actually I guarantee that I'm not going to use these tensors to do autograd later, and as a result, we can do things a little faster.
So there are two things that slow like ostensibly inference only mode code down in Pytorch that inference mode targets.
So the first thing that happens is whenever you do any sort of mutation to a tensor in Pytorch and really whenever you like just allocate any tensor at all, We have some safety tracking for mutation called a version counter.
So what is a version counter in PyTorch? Well, the version counter solves the problem that is pretty common, which is let's say you have a tensor and you need to save its value for later.
Well, tenses are large, and so we don't wanna make copies of them.
So we just save that tester directly.
What if someone along with time when you saved it for, say, backwards.
That's the most common case version calendars are used for.
And when you actually use it, when you do the backwards commutation, someone goes ahead and modifies the sensor under you.
Well, that's great.
It turns out all your, you know, automatic differentiation isn't gonna work.
You're just gonna get wrong gradients in this situation because someone, you know, hunkered about this value and you were expecting the old value prior to the mutation to be the one that you were gonna use for your backwards formula.
So because this can, you know, basically result in silent incorrect results.
Like, you have no idea that things have gone wrong, but things have gone wrong, we have a mechanism called version counters which help us detect when mutations have happened.
The mechanism is pretty simple basically.
We associate every tensor with a version when you mutate the sensor, we update the version, and whenever we save a sensor for backwards, we look at what the current version was and say, okay, whatever this version is, when we look at it again later in the backwards, you have to, you know, have the same version that you had when you saved it.
So if there was a different version, we would just raise an error and say, hey, someone mutated the save transfer backwards.
Uh-oh.
Alright.
So that means that we have to do a bunch of, you know, work.
Right? So we have to allocate these version counters, we can't actually store them directly on the tensor because remember mutating a tensor or mutating a view of a tensor hey, these, you know, are the same thing.
So we need to make sure you get updated in both of these cases.
So it's not something you can store in the tensor directly.
And it also isn't something you can store in the storage, if you know what that is, for very complicated reasons involving detached.
So these are actually like separate heap allocated counters that we keep around and you have to allocate them and you also have to do the reference count bumps on them.
And these these version counter bumps, sorry, not reference count bumps, version counter bumps.
And we have to do these bumps autonomously because there might be a mutation from separate threads.
So that also leads to cost.
Right? It leads to having to do all these extra operations.
So can we get rid of this when there's no requires batch true anywhere in your program.
The answer is no because you don't know if in the future someone is going to use this tensor to actually save it for backwards because it's gonna be used with some of the requires gratitude thing.
So we need to know ahead of time that you know, hey, this is gonna be a tensor that is never ever going to alias with a tensor that is gonna be saved for backwards.
The second thing that we have to do ahead of time is something called view tracking.
So what is view tracking? Well, let's just think about how views work in piper.
So If you've read my blog post about, you know, basic concepts in Pytrols, you may know that Pytrols tenders are strided.
And so if I wanna take a view on a tensor, I can just, you know, allocate another tensor, share the data and just, you know, record, you know, what the offset should be and, you know, whether or not I'm gonna, like, inflate my strides or anything like that.
And this is pretty cool and ordinarily you would think that when I do a view on an operation, that's the only thing I need to do.
Well, unfortunately, in the presence of automatic differentiation, that's not enough.
And the case that causes problems is what if you take a view from a tensor? and then you mutate the view with another tensor that requires gradients.
Let me say that again because it's a little bit of a complicated example.
have a tensor, take a view of it, you mutate the view with a requires gratitude tensor.
So what's very interesting happens in this situation, which is that if you then go back to the base tester and you use it as part of some commutation, that base tester now requires grad equals that requires gratuitousness of the input mutation on the view infects the base tensor.
And if you think about why this might be the case, it makes sense because hey, you know, I have this thing and I need to keep track of all uses of it because, you know, I wanna differentiate on it.
And, you know, if I mutate it into the view, it is gonna like implicitly show up in the base.
And so if I make uses of the base that end up contributing to my loss, Well, those also count as, you know, uses that I have to, you know, count towards, you know, when I do automatic differentiation in this case.
And so just recording, you know, the storage and the strides and the offset in the tensor when we do views isn't enough.
We actually need to record some extra view metadata so that we can make this situation work.
So I've covered the two situations where we need to do this extra work.
So one is in place updates to do version counter bumps.
And the second is view metadata tracking.
And if you were thinking back to the original motivation for inference mode, Well, hey, you know, these are very obscure situations and if I'm just running inference on my tensors, you know, I don't expect any of these things to actually matter.
So inference mode is the way for the user to tell, hey, I am going to guarantee you that I am not gonna do any of these naughty things and then I can just skip doing version counters, so I I just won't allocate the version counters at all.
I won't do version counter bumps on my tensors.
and I'm just not gonna do any of the view metadata tracking.
I'm just gonna, you know, leave it all alone.
And then, you know, my code will run faster as long as I'm not using it for AD.
So that doesn't sound too hard.
Right? Just put in a bunch of if statements and, you know, or, you know, like because we've talked about the dispatcher.
Right? Oh, do some fancy dispatchers stuff.
Just make these things not get run-in those cases, but there's a problem.
The problem is we don't actually want to have our users pinky promise us that they're gonna handle everything correctly because we don't actually trust our users to do things correctly.
you shouldn't either.
I wouldn't trust myself to get this thing right.
I'm worried that I am gonna accidentally use one of these tensers in autograph later and everything's gonna blow up and, like, I'm gonna be sad.
So those sort of magic sauce and what sort of took us a long time to sort of get Inference mode working was how do we do this safely? Now to say, how can we let the users say, I promise not to use these things for autograph and then actually hold the user to this promise so that if they actually do use it for Inference mode later, If they use an inference mode sensor in automatic configuration, we actually give a proper error message in this case.
And so I'm just gonna describe a little bit about how we do this.
And, you know, if you wanna actually see the details We've got a very nice RFC, co authored by Aileen and Me, and you can read that for all those sort of niggity details of how everything works.
But there's two basic things that we need to do.
So the first thing that we need to do is we want to get rid of version count owners.
Right? We wanna get rid of the need to track when mutations happen.
And so in order to verify that, you know, this never actually causes problems, for automatic differentiation, we need to enforce some sort of invariant that says, oh, yeah, you know, one of these tensors that doesn't record version counters.
you're not allowed to ever actually try to use the version counter to enforce safety because that's a play that's a place where the system could go wrong.
So in other words, we have a no aliasing requirement.
The no aliasing requirement says that any tensor that doesn't have the version counter we're actually gonna just refer to these as inference tensors because they're just tensors that happen when you do inference mode.
Right? You just don't allocate version counts for them.
Any inference tensor must not alias with any tensor that is saved for backwards.
So how do we actually do this? Well, you know, we take an inference tensor, we say, okay, there's no version counter on it.
Whenever we make aliases to this tensor, we also need to make sure these are also inference tensors.
because, you know, hey, it's an aliasing requirement.
Right? Like, you know, just because you take a view of a sensor doesn't mean you can say that because if you mutate that, well, you know, it still affects the view of the sensor.
And then we just say, okay, any inference sensor is not allowed to be saved for back And so there's one place we have to write this check, which is namely when we save variables for tensors.
So the no aliasing invariant involves basically setting up this dynamic alias analysis that just says, hey, this is a class of sensors, these inference sensors which are guaranteed not to alias with AD, and we only have to check one in place to make sure this actually happens.
And so that's very nice and not too hard to implement.
Second part is view tracking.
Right? So what do we do if we, you know, don't track the view metadata in a situation.
And this one's actually not so hard.
We basically just say, okay.
we don't record the view metadata for these tensors.
And now we need to Sorry, I I said this one's not so hard, but this one's also tricky in its own way.
So naively, what you'd expect you'd be able to do in this situation say, okay, I'm just not gonna record the view metadata.
And then if I ever do something to a tensor that, you know, might require the view metadata, I just raise an error.
Does that work almost, but there's one problem.
And the problem is if you have a base tensor, and you mutate it with something that requires credit equals true, ordinarily your views also become requires credit equals true.
Right? The the the the flow goes both ways.
Right? Like, if I put in some data that I need to track gradients for, then all the views also need to track gradients as well.
And in the case of the base sensor, I don't act actually know if I've recorded the view metadata or not in the situation.
So what we do is we just say, okay, well, these inference tensor things, you know, the tensors that were allocated in her influence mode, you're not allowed to mutate them outside of influence mode.
And that just sort of, you know, with a very heavy hammer, prevents this sort of situation from causing a problem.
So that's what Inference mode does in a Not Shaw.
It says, okay.
When you're inside Inference mode, you know, we allocate these inference sensors.
These inference sensors do less work.
They don't track versions and they don't track view metadata.
And once you have this situation, you just have a bunch of extra checks, a bunch of, like, sort of restrictions on how you can use these tensors outside of inference mode that sort of guarantees that you can't actually observe that you fail to record all this information.
You'll just error in those cases.
So we've been deploying this to a bunch of places.
There's this old r a i i car guard called auto non variable type mode.
It didn't make any sense It just happened to make people's code run faster, but it didn't do any error checking.
And we moved been moving people over to use inference mode in a situation.
Actually, that's all idling stuff.
She's been very like a trooper moving all of our mobile stuff over.
It's been quite an adventure because There's a ton of places that only do inference.
Like, ever try to debug a Pytorch problem on oculus.
Yeah.
Me neither.
good work.
So that's everything I had to say about Inference mode today.
Right now, it's only available from c plus plus, but we'll we'll be adding a Python API for it very soon.
So that's all I wanted to say for today.
See you next time.
.
