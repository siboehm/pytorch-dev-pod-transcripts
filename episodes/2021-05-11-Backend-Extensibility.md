---
layout: post
title: "Backend Extensibility"
date: 2021-05-11
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Backend Extensibility

Hello, everyone, and welcome to the Pytruch Sturm podcast.
Today, I wanna take a gambling journey through what we like to call in Pytorch as back end extensibility.
What do I mean by back end extensibility? Well, Pytorch you know, as a project has a number of things that it's supposed to do.
And one of the things that it's supposed to give is GPU acceleration.
Right? Because if we didn't have GPU acceleration, you could just use Numpy and do a lot of things you wanted to do there.
But GPU acceleration means that you can take your same pilot program and run it not only on CPU, but also on CUDA at the same time.
And so we call these things back ends.
At the very beginning of the Pirate project, CPU and CUDA were the only backends that were available.
And if you go back to the Lewisworks days or also further back to the torch seven days.
Right? There was the TH library and there's the THC library.
And those are really the only backends in town.
So everyone sort of used one or the other.
And when Patrick was initially released, that was still the same deal.
We had a CPU, and we had a CUDA, and that's all that happened.
then over time, people, you know, came to us and they're like, oh, I've got some hardware or I've got some other back end.
And I've also like to use it with Pytorch in the framework.
And we started working on making it possible to add more and more back ends to Pytorch.
So that's what I mean by back end extensibility.
And so where is Pytorch when it comes to back end extensibility? Well, let's dig into it.
So the first thing to really know about PyTorch is that from the beginning, it was designed for CPU and CUDA.
So if you have something that looks a lot like CUDA as the back end you wanna do, things are gonna work out okay for you.
And so a really good example of something that's really like CUDA.
In fact, so like CUDA that, like, it's actually just freaking transpiling CUDA kernels into their own kernel language.
is the AMD Hip Rocum project.
So, you know, CUDA is an Nvidia invention and it's only targeted at NVIDIA GPUs.
AMD also produces GPUs and for the longest time they didn't have any general purpose programming capabilities.
Well, Rocum is AMDSec for doing doing general purpose programming on their GPUs.
And the way they set things up was they were like, okay.
While CUDA has a, you know, decade long head start and, you know, building a software stack, really one of the, like, key advantages of, you know, being on NVIDIA hardware.
Let's just try to make use of as much of it as possible.
And so the way ROCCM works is that they have a language for kernels that is basically the same as CUDA.
So they like copy paste it as much of the language as possible.
And so when you want to write a kernel, in PyTorch, you write a CUDA kernel, and then in our a m d hip rock and build, we actually transpire all this kernel by just doing a bunch of regular expression it replaces really.
We call it hipification.
but it really is just a bunch of, you know, search and replace surround strings.
We typify it into a hip kernel, and we just directly go ahead and compile that and that's what, you know, you actually run if you're running your PyTorch on an AMD GPU.
This actually works, by the way, Like, you can get one of the few there aren't that many GPUs that AMD releases that support Rockom.
But if you have one of those GPUs, you can run your patronage programs on it.
We did make some weird choices with Rocum because this was one of the first first external back ends that we added to Pytorch.
And one of the weird choices we made was that so remember, Pytorch is only CPU and CUDA.
And so while Rocum is like this thing, but no one is writing code so that it works with Rocum.
what they just did, they were like, okay, we're just gonna, you know, not rename any of the user facing interface.
So if you want to put your, you know, tensor into rocket memory, just call CUDA on it with our special AMD build of PyTorch, which is mutually exclusive from the regular Nvidia builds the PyTorch, and then that'll put it on the Rockom GPU.
Kinda goofy, worked out okay for them, it was easy for them to reuse all of our tests.
Right? because all our tests were just written assuming that CUDA was a thing and, you know, not really a big deal.
kind of annoying from my perspective.
I really hate that it's CUDA masquerading as it's sorry.
It's hit masquerading as CUDA, and I'd really like them to fix it someday.
I haven't got been able to get them to fix it.
But so that's how they do it.
And because Rocum is so so similar to CUDA, like, literally, like, most things that CUDA provides like streams and, you know, current device and, you know, like CUDA and N, they all translate into the hip world.
So it wasn't too hard.
It's not too hard right when, like, someone is doing the API exactly the same way.
The sort of next example of device extensibility that sort of lives in Piper Jaffray's history is our XLA integration.
So XLA, if you're not aware, is Google's underlying compiler for TensorFlow.
So, like, TensorFlow is the front end you can write your neural networks in, but then x l a is this compiler that can take in a high level i r and then compile it.
And so for example, JAKKS, the new darling of, you know, Google's research researchers.
Jack's also targets XLEA and, you know, he doesn't have to share any tensorflow code but it just, you know, uses the underlying compiler.
So XLEA is pretty cool.
If you wanna run your code on GPUs, Google's hardware for deep learning, XLEA is the only game in town.
And so we wanted to also make it possible to run PyTorch programs on TPUs, and as a result, we invested in an excellent integration.
Now, XLEA is a lot different from CUDA and ROCOM.
Right? So CUDA, if you remember from my podcast about everything else, just, you know, enough to know enough about CUDA to be dangerous.
CUDA is an asynchronous programming model.
Right? So you, like, have a bunch of kernels.
You call into CUDA and CUDA goes and says, okay.
Well, these are the things I'm gonna run as you told me too.
Well, XLE is nothing like that.
Right? XLE is a graph mode compiler, it expects to be given a graph of high level IR, and then it will actually optimize it and run it for you.
we had to do something completely different for XLEA and this is what we did, what was like we added a new XLEA type because similarly to hip, we, you know, wanted the main code Sorry.
Not similar to hit.
XLEA has a bunch of integration that needs to work with the XLEA code base, and so we wanted to let lift have let that live out of trees.
So what we did was we put in a bunch of hooks in Pytorch core to sort of make this all work.
And one of the things we did is there's an LAXLA device similar to how there's a CPU and coded device type.
So you have to go and send a PR to Pytorch core and say, hey, you know, can I have this device type? So you put in this device type, And then we have this dispatcher thing, which I also had talked about in the previous podcast.
You know, this dispatcher thing that is entry point where for any of the operators that are defined in native functions at yamal, you can define your own implementation of it register it at runtime.
So, like, literally, you have a dynamic library and has a bunch of static initializers.
And those static initializers are registrations.
there's a very user friendly API modeled off to Pi bind eleven that you can use to do this.
So you register those things and then whenever there's a tensor that is an XLA tensor, when you call any code in PyTorch, it will it will hit the dispatcher.
You'll see that, oh, this is an XLA tensor.
and then it will route to the dynamically loaded XLEA implementation that does whatever you want.
And then XLEA itself because, you know, it's a graph mode thing actually doesn't do any computation, it just goes ahead and builds a computation graph.
And so at the front end, there's some stuff you have to do differently.
Like, there's a special set of optimizers we should know how to deal with the fact that x l a computation is lazy and not eager.
But like x l a was sort of the first like, actually usable external back end that we developed for.
And we did a only so so job in porting them in their endeavor.
So actually, you know, it turns out there's a lot of boilerplate you have to write when you want to support for an external back end.
And also, XLEA doesn't support all operators in Pytron.
Right? Pytron is a lot of operators.
And XLEA you know, well, XLA is cool and there's a lot of operators that it just doesn't support or, you know, like they just didn't have time at support for.
So XLE also has this mechanism for allowing for fallback to CPU where it's like, okay, you're running your cartridge program on XLE.
and then you get to some operator that XLEA doesn't know how to do.
So XLEA is gonna go ahead and compute the actual output for you, given the XLEG graph that was at hand, and then with a CPU Tensor, fall back to calling the regular PyTorch CPU kernel, and then, you know, doing it back to XLEG.
It's kind of a question whether or not this is a good idea to do by default or not because these are like terrible performance clips.
But it's really useful if you just wanna figure out if your program is gonna run or not.
Right? Just like being able to move in between these ways.
So because there's kind of a lot of like infrastructure you have to write.
XLA actually went ahead and built a mini code generator, like the it's this python script that gets run.
during the build process of XLA that actually generates all the code that registers to Pytorch.
And XLA like predates on sort of nice registration API.
We had a not so nice registration API before, and it was very hard to use.
And so XLA has this code and it does all this stuff and it's not so nice.
So actually, Brian Hirsch, one of the members of the composability team, has recently been working on sort of revamping exlates cogeneration and letting it live in Pytorch as a thing that external backbones can use.
When they wanna, like, you know, plug into our system, and get all the niceties, like, you know, fall back to CPU in this situation.
So rewinding a sec.
So what is adding a new back into pixels look like in the x-ray universe.
Well, you you need to, you know, first send a PR to pixels main repository being like, hey, hey, hey, I got this new device called XLA.
You know, I need Pytruxure to know about You also have to tell Pytruxure about this dispatch key thing.
Dispatch key is the thing that actually, like, you know, we do the virtual call based on.
It's different from the device type because not all device types have dispatch keys and Also, we have a bunch of concepts that aren't device types like v mapping and meta tensors.
Actually, meta tensors count as device.
and, like, autograph.
And these things aren't really devices in their own right.
So dispatch key is this, like, generalized idea of all the things you might go to.
You know, to go ask for dispatch key to be added.
And then there's a little bit of Python binding code, which we never, like, wrote in generic way.
So you gotta go edit those parts.
But once you do all those things, you basically don't need to do anything else in Pytosh Court.
Right? Because there's this virtual table that you can manually program in using torch library macros.
And this is what X Lite does.
And it's it's a little not so nice to do this directly, and so people have resorted to cogeneration.
to actually manage these things.
So this is like basically the current state of the art in external back ends.
I I guess something that's a little good to talk about is, like, what are some of the challenges of doing National Back end in this way? Because we've actually had a bunch of people try to actually go on to this treadmill.
And you'll see why I call it treadmill in a moment.
For example, ML compute from Apple and Intel as well.
So some of the reasons why this is a little difficult.
So one is that, you know, PyTorch cares a lot about backwards compatibility.
but only for our end users and not our back end extenders.
So here's an example of a backwards compatible chain you can make.
which is say there's some function and we wanna add some new functionality to it, we could add a new argument to that function and give it a default value.
So, you know, from the perspective of someone using Pytrux from Python, this is perfectly backwards compatible because, hey, you know, like, if I am not passing this argument, it'll get defaulted, and then I will ostensibly get whatever old behavior I had.
Well, that's not the case.
when you're doing a back end because, well, when you, you know, have to register one of these functions that knows how to process this operator, you have a problem.
Right? Like, this operator is now trying to give you extra arguments, and you're like, oh, well, my old operator implementation only knows how to handle three arguments, not four arguments.
What do? In principle, this isn't actually a busy breaking change.
Right? We could, like, somehow detect that the user gave us the defaulted argument and then, you know, call your old function without that argument in that case.
And if they give a non default value for the new argument than the error in that case.
But it's kinda hard to do this in c plus plus only.
So, you know, this is not something that, like, we work.
And the upshot of this is that, like, if you want to like do a back end, you're actually gonna have to do a lot of work keeping all of your, you know, understood functions up to date with with the changes to all the operators.
because we keep adding new operators.
We keep adding new knobs on operators.
And So it's it's kind of a treadmill keeping up to date.
By the way, XLA can keep up to date because we actually have it included as part of the build system in PyTorch So, like, whenever you're working on a new operator in Pytorch, XLA will tell you if you broke XLA.
And then, like, only do the heroic efforts of, you know, Jack Sow and the rest of the XLE team, does this actually work okay? Because, like, you accidentally break the XLE test.
You're an average fighter's developer.
You know nothing about XLA.
You can just sort of send up the bat signal, and they will make the compatibility patch to, like, get XLA going.
What I've heard, some other people I've done when they're extending the back end in this way is they, like, don't bother.
And so every release they, like, try to catch up.
And there's, like, a ton of stuff then.
This is not so great.
I'm hopeful that Brian Hersha's work on a co gen for actionable back ends can make this easier because there's some things that we just technically can't do in pure c plus plus.
but are easier for us to do in Python.
But this this code is still being in the process of being landed for XLA.
It's really close.
We we actually tried landing it a few days ago, and it got reverted because it broke something.
But not not for a good reason.
Like, it's been passing passing half of the CIs for a good while now.
So I'm out of time.
There's probably more things about back end extensibility that I should talk about, but I'll save them for another podcast.
See you next time.
.
