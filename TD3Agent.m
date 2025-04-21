numobs = 4;
obsinfo = rlNumericSpec([numobs,1]);

actinfo = rlNumericSpec([1,1],UpperLimit = 20,LowerLimit = -20);
lrq = 1e-4;
lrpi = 1e-4;
obsin = [featureInputLayer(prod(obsinfo.Dimension),Name = 'obsin')];
actin = [featureInputLayer(prod(actinfo.Dimension),Name = 'actin')];

commpath = [concatenationLayer(1,2,Name='concat')
            fullyConnectedLayer(128)
            reluLayer
            fullyConnectedLayer(128)
            reluLayer
            fullyConnectedLayer(1)];
actpath = [ featureInputLayer(prod(obsinfo.Dimension),Name = 'obsin')
            fullyConnectedLayer(128)
            reluLayer
            fullyConnectedLayer(128)
            reluLayer
            fullyConnectedLayer(1)
            ];

cnet = dlnetwork;
cnet = addLayers(cnet,obsin);
cnet = addLayers(cnet,actin);
cnet = addLayers(cnet,commpath);

cnet = connectLayers(cnet,'obsin','concat/in1');
cnet = connectLayers(cnet,'actin','concat/in2');

anet = dlnetwork(actpath);
%%
tiledlayout(1,2)
nexttile
plot(cnet)
title('Critic Network')
nexttile
plot(anet)
title('Actor Network')

%%
critic1 = initialize(cnet);
critic2 = initialize(cnet);

q1 = rlQValueFunction(critic1,obsinfo,actinfo, ...
    "ObservationInputNames",'obsin', ...
    'ActionInputNames','actin');
q2 = rlQValueFunction(critic2,obsinfo,actinfo, ...
    "ObservationInputNames",'obsin', ...
    'ActionInputNames','actin');



actor = initialize(anet);

pi1  = rlContinuousDeterministicActor(actor,obsinfo,actinfo, ...
        'ObservationInputNames','obsin');
UseGPUCritic = true;
if canUseGPU && UseGPUCritic    
    q1.UseDevice = "gpu";
    q2.UseDevice = "gpu";
end

UseGPUActor = true;
if canUseGPU && UseGPUActor    
    pi1.UseDevice = "gpu";
end

if canUseGPU && (UseGPUCritic || UseGPUActor)
    gpurng(0)
end
% criticOptions = rlOptimizerOptions( ...
%     Optimizer='adam', ...
%     LearnRate=lrq...
%     );
% 
% actorOptions = rlOptimizerOptions( ...
%     Optimizer='adam', ...
%     LearnRate=lrpi...
%     );
% 
% agentOptions = rlTD3AgentOptions;
% agentOptions.MiniBatchSize = 512;
% agentOptions.DiscountFactor = 0.99;
% agentOptions.CriticOptimizerOptions(1) = criticOptions;
% agentOptions.CriticOptimizerOptions(2) = criticOptions;
% agentOptions.ActorOptimizerOptions = actorOptions;
% agentOptions.SampleTime=1/60;
% agentOptions.ExplorationModel.StandardDeviationMin = 0.05;
% agentOptions.ExplorationModel.StandardDeviation = 0.1;
% agentOptions.ExperienceBufferLength = 5e5;

agentOpts = rlTD3AgentOptions( ...
    SampleTime=1/5, ...
    DiscountFactor=0.99, ...
    ExperienceBufferLength=5e4, ...
    MiniBatchSize=50);

for idx = 1:2
    agentOpts.CriticOptimizerOptions(idx).Optimizer='adam';
    agentOpts.CriticOptimizerOptions(idx).LearnRate = 1e-3;
    agentOpts.CriticOptimizerOptions(idx).GradientThreshold = 0.5;
    agentOpts.CriticOptimizerOptions(idx).L2RegularizationFactor = 1e-4;
end

% Actor optimizer options
agentOpts.ActorOptimizerOptions.Optimizer = 'adam';
agentOpts.ActorOptimizerOptions.LearnRate = 1e-3;
agentOpts.ActorOptimizerOptions.GradientThreshold = 0.5;
agentOpts.ActorOptimizerOptions.L2RegularizationFactor = 1e-4;

agentOpts.ExplorationModel.StandardDeviationMin =  0.025;
agentOpts.ExplorationModel.StandardDeviation = 0.05;
agentOpts.TargetPolicySmoothModel.StandardDeviation = 0.05;
agentOpts.PolicyUpdateFrequency = 3;
agent = rlTD3Agent(pi1,[q1 q2],agentOpts);
