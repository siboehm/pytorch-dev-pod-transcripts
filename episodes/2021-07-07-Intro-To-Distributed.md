---
layout: post
title: "Intro To Distributed"
date: 2021-07-07
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Intro To Distributed

Hi, everyone, and welcome to the Pytorch dev podcast.
Today, I am doing something a little special, which is that I have Shen Lee from the distributor team over at Pytorch.
Here to come talk to us about titles distributed.
Shane, do you want to introduce yourself? Hello, everyone.
This is Shane.
I work on titles distributed package.
Super happy to be here.
Alright, Sean.
So I just wanna get started.
Can you just explain to us, you know, what distributed training is and, you know, why it's so important for PyTorch? Of course.
I would say distributed training is using multiple GPUs or machines to collaboratively train the same model.
By the way, this is just my personal view, not an official performer definition.
I can try to elaborate on that statement.
Yeah.
Tell me tell me a little more.
Yeah.
Let's let's start from the motivation side.
Like, why do we need multiple GPUs for machines to train my model? It is because driven by the hypothesis in deep learning applications.
People are using larger and a larger data sets to train larger and a larger models.
It's possible that the data does not fit in one machine, or maybe the model does not fit in one machine.
Or even if they both can't fit in one machine, you might still want to leverage more resources to finish training within a shorter period of time.
So that's why we might wanna use multiple GPUs or machines to train a model.
Okay.
So let's say that, you know, I happen to have a giant cluster of machines in my back pocket and like I wanna use them all How do I go about doing that? So there are like a lot of different tools out there.
It depends on like whether you are a framework developer or application developer.
If you are application developer, you can choose the right tool in PyTorch.
There are DDP RPC pipelines that are If you are a framework tool, then there are, like, a lot of different things you need to consider to make sure that the distributed training can work efficiently.
So when going beyond one GPU and one machine, the communications are, like, inevitable.
And communications are really very slow.
And so if when you're working on that, if a communication blocks computation, there will be, like, low device utilization and has low efficiency.
So that's like one challenge you need to handle if you wanna work on distributed training.
Would you say that dealing with the cost of communicating over nodes is the biggest problem when working on distributed training? I guess that's one of the biggest problem because the main delay of distributed training comes from two sources.
one is, like, computation and the other is communication.
And there are actually a lot of tools just try to handle that.
One fortunate thing is that since communication and computation are using different resources, so they can actually overlap.
They can basically wrong concurrently.
So that's like one benefit we can try to explore to speed up things.
So earlier in the podcast, you told me that if you were a user of distributed, you had a bunch of options for what you could do.
For example, you could use RPC, you'd use GDP, And also if you're a library author, you could use some other options.
So what are like, I think one of the things that people find the welding bewildering about distributed is how many things you can do? Like, how many different options for setting things up? Could you just, like, tell us at a high level, like, how these all get put together, how you decide to choose one or the other? Oh, sure.
Yeah.
There are, like, a lot of particular options.
for data parallel, they are the data parallel.
Hang on a sec.
So tell tell us what data parallel means in this context.
Oh, sure.
with Makena data parallel training, each GPU holds the replica of the model and consumes a split of the inter data.
and models are synchronized using communications.
So, basically, models are replicated and data is charted.
And the entire model gradients and the parameters are communicated across replicas to make sure that they are synchronized.
So this is the, like, vanilla data paradigm.
And vanilla model paradigm is the opposite where data will flow through all devices and model is sharded across devices.
And the communication is only responsible for transmitting the activations and its corresponding gradients at model sharding boundaries.
And of of course, there are, like, more complicated data parallel and multi parallel schemes.
And there are also like hybrid parallel schemes that combines both data and a multi parallelism.
So that's like a very high level description of data and model parallelism.
And beyond that, there are, like, advanced versions.
Like, in in Python's DDP, it's a vanilla data parallel plus some optimization.
I can try to go a bit deeper into that.
So as I mentioned, like communication might be one of the main thing people need to do with if if you're working on distributed training.
And one, like, natural thing to optimize distributed training is try to overlap communication with computation as much as possible.
because communications, so communications are the main sources of delay in distributed training.
And overall, the communication delay will also, like, grow or waste a cluster size.
And since they are using different type of resources, it's often possible to run them concurrently.
And actually, existing distributed training technologies, like GDP, are using such optimizations.
What DDP does is that, say, when you are synchronizing the gradients of layer i in the backward path, you can just go ahead and do the computation on a layer i i minus one to compute the gradients.
in in that way, the computation and the communication can overlap.
So are you saying that you, like, sort of, there's a train of computation going on So at any given time, each of the layers is processing a different set of data in this situation? Not exactly.
So, yeah, we can try to open up the background pass a bit and see how the communication got plugged into background pass.
So in the backward pass, we have a layer we have a model multiple layers.
Right? And the backward pass gonna flow from the last layer all the way back to the first layer.
and it's like a stack of layers.
And then the communications responsibility here is trying to make sure that the gradients on all the model replicas are the same after the back over the past.
So how we can do how how we can do that? Like, what a solution start, which is to run the local autograph engine.
and making sure all the ingredients are ready on each process, but they're gonna be different because they are consuming different data, the input data.
And then we can basically around, say, an hour radius to communicate the ingredients to making sure that making sure that they are the same.
But this is gonna be slow because you see that there's no overlap at all for computation and communication.
Because I'm waiting for the gradients to get completely computed before I start the next batch of processing.
Is that right? Yeah.
Exactly.
And in this case, basically, the GPU can be busy.
for a while to do the computation.
And then when you start a communication and then the GPU computation resource on the GPU is gonna be idling.
Just waiting for the communication to finish.
So basically, at any individual point of time, there's only one type of resource that is busy.
And this is bad.
This is what we are trying to avoid.
Okay.
So how do I fix it? So in DDP, the solution is that we are organizing the gradients of the model into buckets.
So for example, if you have twenty layers in your model.
It's possible that you organize, say, last five layer into the into one bucket and then the next five layer into another bucket.
And then when you finish computing the gradients for the last five layers, you can put the grid you can put the gradients of the of those five layers into the into bucket.
and then kicking off the communication of that bucket.
And at the same time, you can continue to do the computation for the ingredients for the next five years So in this case, the computation of the next five layers and the communication of the last five layers will be wrong in parallel.
So to put it in other words, the the thing that happens, right, is that although your model has a lot of parameters, we manage to compute some of these parameters before other parameters.
And so if we can go ahead and start updating those parameters that we're already done computing before we're done with everything else, we can get ahead of having to wait for everything to be done and then doing the synchronization in that case.
Yeah.
Yeah.
Exactly.
That's a very great great summary for for data parallel.
And going back to your question, like, what are the, like, options of distributed training in the in the market? And there are, like, other things, like, pipeline parallel, shorter data parallel.
They actually many of them are actually exploring the same basic idea of trying to overlap.
communication with computation.
Just like what DDP does, but they do that for different things.
Like, for pipeline parallelism.
What pipeline parallelism do is that for every minute batch, you can divide one minute batch into into multiple micro batches.
And then the model is basically chartered across multiple devices.
and then gonna fit the first micro batch into the first model chart.
And then when when finished the competition on that, you're gonna move the activation from the first device to the second device.
And then you can feed the second micropash into the first model chart and etcetera.
So, you know, the pipeline basically can run.
And in this way, it is able to basically keep module devices running in parallel.
And, also, when you are doing a competition, say, microbatch, i, and you can concurrently launch the communication for the activations generated by MicroPash i minus one.
So In this way, the pipeline can work and make computation and computation and computation overlap.
But it it it actually based on a a similar idea of trying to overlap things.
So to summarize, it seems that, like, first, at a high level, you have to decide what you're gonna paralyze over.
Are you gonna paralyze over the data or you're gonna paralyze over the parameters.
But even once you've made that choice, there are a bunch of optimizations you can apply for overlapping computation and all those optimizations result in tons and tons of possibilities for how you can go about doing your distributed training.
And I'm guessing, like, it's different depending what model you're trying to train as well.
Right? Right.
Right.
I think that's a correct statement.
And one thing I wanna add is that yeah, initially, you need to make a decision whether you need data parallelism or model parallelism.
And, usually, when models are small, data parallelism will be sufficient.
And when models are large, you usually want to combine mono parallelism with data parallelism.
because the data set is really very large.
And if you just have, say, one more rapid ride than tire cluster, it it is possible to get get up to, like, higher speed but usually having a higher data parallel with what also have you to speed up training a lot.
Alright.
So I wanna turn our attention now towards the state of distributed in Pytorch.
Because I think the discussion that we just had could apply to a distributed framework anywhere in, you know, like, TensorFlow or PyTorch or any of the other deep learning frameworks.
So what is different about nitrogen distributed? Like, how did Nitrogen distributed come to be the way it is today? We started working on Nitrogen distributed.
I think since twenty nineteen twenty nineteen.
And the first feature we developed is distributed data parallelism.
And at that time, data parallel data parallel is the most dominant distributed training.
technology.
And so far, it's still the most dominant distributed training solution.
And later, when with the advances in the community, and people have started to deal with larger and other models, we started to realize that when the data parallelism is now sufficient because the model cannot it's possible that model won't be fitting into one GPU and one machine.
So we started to think about, oh, we need to add a model paradigm feature into the package.
And that's when we started to thinking about, oh, how do we do a generic mono pairing solutioning? How do we add a generic mono pairing solution in high touch? and come up with the idea of adding say RPC remote procedure call.
Basically, what we are trying to do here is to making sure that everything that user usually used for, say, forward, backward and optimizer in local training can be represented using the distributed APIs.
And that's also what we are focusing on when developing the RPC package.
So, basically, with the RPC package -- Mhmm.
-- what you can do is you can grab some part of the model into your functions and use the RPC to launch that functional remote worker.
And RTC will be responsible for things like serializing and serializing the sensors and also take care of the distributed autograd.
And also, there is a counterpart for a counterpart of optimizer, for a local optimizer where this really option either gonna automatically reach out to you or the participating processes and getting the parameters updated.
So that's like the second step we take.
after data parallelism.
And after that, we're also starting to build, like, more and more higher layer features on top of RPC in cartridge because RPC is like a very low level raw API, which is black spoke, but it's not that easy to use.
If you wanna use RPC, you will have to do things like decomposing model and write a lot of the code to make it work.
And ideally, we wanna we wanna make sure that when you have a model that can trend locally, the same model, maybe with a a larger size, can trend on the spirit environment as well.
So we need to we need to have a higher level APIs to make that happen.
And things we added so far are, like, pipeline parallelism And we're also working on things like intra layer sharding to make sure that you you can not only sharding the model based on the operator boundaries, You can also stay short one operator and clear that across multiple processes.
One of the themes in my podcast has been that Pytorch, you know, originally was designed as an eager mode framework.
And so whenever we build any features, you know, we always try to figure out how it can work on your mode first, and then other modes of operation come later.
And, you know, some of the things you described, right, like building the higher level API for distributed.
You honestly have a harder job than some of your competitors who, you know, can assume there's a graph representation because you need to work with ear mode PyTorch.
Yeah.
That is true.
And and actually, that that's the ongoing discussion in the team.
We are thinking like which we are thinking with we are collaborating with the compiler team, and we are thinking about, like, which layer that we we should be able to, like, extract the graph from the from the forecast.
And based on the outer divide, the model chart to to divide model and do the model placement.
So far, we don't yet have a great answer.
Like, things like touch FX, and Jet IR can also can definitely be helpful.
But we haven't decided yet whether those should be the solution where we do.
on top of or we need something else.
Alright.
Well, thank you very much, Sean, for joining me.
I'm hoping that we can do more of these interview style podcasts in the future.
Thanks, everyone, for listening.
Talk to you next time.
Talk to you next time.
Bye.
.
