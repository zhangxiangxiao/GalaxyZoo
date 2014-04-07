--[[
Main driver for Galaxy Zoo
By Xiang Zhang @ New York University
--]]

-- Necessary functionalities
require("nn")
require("cutorch")
require("cunn")
require("gnuplot")

-- Local requires
require("data")
require("model")
require("train")
require("test")
require("mui")

-- Configurations
dofile("config.lua")

-- Addtional modules
if not nn.Dropout then dofile("Dropout.lua") end

-- Create namespaces
main = {}

-- The main program
function main.main()
   -- Setting the device
   if config.main.device then
      cutorch.setDevice(config.main.device)
      print("Device set to "..config.main.device)
   end

   main.clock = {}
   main.clock.log = 0

   main.new()
   main.run()
end

-- Train a new experiment
function main.new()
   -- Load the data
   print("Loading datasets...")
   main.train_data = Data(config.train_data)
   main.val_data = Data(config.val_data)

   -- Load the model
   print("Loading the model...")
   main.model = Model(config.model)
   if config.main.randomize then
      main.model:randomize(config.main.randomize)
      print("Model randomized.")
   end
   main.model:type(config.main.type)
   print("Current model type: "..main.model:type())
   collectgarbage()

   -- Initiate the trainer
   print("Loading the trainer...")
   main.train = Train(main.train_data, main.model, config.loss(), config.train)

   -- Initiate the tester
   print("Loading the tester...")
   main.test_train = Test(main.train_data, main.model, config.loss(), config.test)
   main.test_val = Test(main.val_data, main.model, config.loss(), config.test)

   -- The record structure
   main.record = {}
   if config.main.resume then
      print("Loading main record...")
      local resume = torch.load(config.main.resume)
      main.record = resume.record
      main.show()
   end

   -- Initiate the windows of drawing the model
   main.mui = Mui{width=config.mui.width,scale=config.mui.scale,n=config.mui.n,title="Model Visualization"}
   main.draw()
   collectgarbage()
end

-- Start the training
function main.run()
   --Run for this number of era
   for i = 1,config.main.eras do
      if config.main.dropout then
	 print("Enabling dropouts")
	 main.model:enableDropouts()
      else
	 print("Disabling dropouts")
	 main.model:disableDropouts()
      end
      print("Training for era "..i)
      main.train:run(config.main.epoches, main.trainlog)

      print("Disabling dropouts")
      main.model:disableDropouts()
      print("Testing on training data for era "..i)
      main.test_train:run(main.testlog)

      if config.main.test == nil or config.main.test == true then
	 print("Disabling dropouts")
	 print("Testing on test data for era "..i)
	 main.test_val:run(main.testlog)
      end

      print("Recording on era "..i)
      main.record[#main.record+1] = {train_loss = main.test_train.l or 0,
				     val_loss = main.test_val.l or 0}
      print("Visualizing loss")
      main.show()
      print("Visualizing the models")
      main.draw()
      print("Saving data")
      main.save()
      collectgarbage()
   end
end

-- Final cleaning up
function main.clean()
   print("Cleaning up...")
   gnuplot.closeall()
end

-- Draw the graph
function main.show(figure_loss)
   main.figure_loss = main.figure_loss or gnuplot.figure()

   local figure_loss = figure_loss or main.figure_loss

   -- Generate losses
   local epoch = torch.linspace(1,#main.record,#main.record):mul(config.main.epoches)
   local train_loss = torch.zeros(#main.record)
   local val_loss = torch.zeros(#main.record)
   for i = 1,#main.record do
      train_loss[i] = main.record[i].train_loss
      val_loss[i] = main.record[i].val_loss
   end

   -- Do the plot
   gnuplot.figure(figure_loss)
   gnuplot.plot({"Train",epoch,train_loss},{"Validate",epoch,val_loss})
   gnuplot.title("Training and validating loss")
   gnuplot.plotflush()
end

-- Draw the visualization
function main.draw()
   main.mui:drawSequential(main.model.sequential)
end

-- Save a record
function main.save()
   -- Record necessary configurations
   config.train.epoch = main.train.epoch
   
   -- Convert the model to double
   main.model:double()

   -- Make the save
   local time = os.time()
   torch.save(paths.concat(config.main.save,"main_"..(main.train.epoch-1).."_"..time..".t7b"),
	      {config = config, record = main.record})
   torch.save(paths.concat(config.main.save,"sequential_"..(main.train.epoch-1).."_"..time..".t7b"),
	      main.model.sequential)
   main.eps_loss = main.eps_loss or gnuplot.epsfigure(paths.concat(config.main.save,"figure_loss.eps"))
   main.show(main.eps_loss)
   local ret = pcall(function() main.mui.win:save(paths.concat(config.main.save,"sequential_"..(main.train.epoch-1).."_"..time..".png")) end)
   if not ret then print("Warning: saving the model image failed") end

   -- Revert back the model
   main.model:type(config.main.type)
end

-- The training logging function
function main.trainlog(train)
   if config.main.collectgarbage and math.mod(train.epoch-1,config.main.collectgarbage) == 0 then
      print("Collecting garbage at epoch = "..(train.epoch-1))
      collectgarbage()
   end

   if (os.time() - main.clock.log) >= (config.main.logtime or 1) then
      local msg = ""
      
      if config.main.details then
	 msg = msg.."epo: "..(train.epoch-1)..
	    ", rat: "..string.format("%.2e",train.rate)..
	    ", obj: "..string.format("%.2e",train.objective)..
	    ", dat: "..string.format("%.2e",train.time.data)..
	    ", fpp: "..string.format("%.2e",train.time.forward)..
	    ", bpp: "..string.format("%.2e",train.time.backward)..
	    ", upd: "..string.format("%.2e",train.time.update)
      end
      
      if config.main.debug then
	 msg = msg..", bmn: "..string.format("%.2e",train.batch:mean())..
	    ", bsd: "..string.format("%.2e",train.batch:std())..
	    ", bmi: "..string.format("%.2e",train.batch:min())..
	    ", bmx: "..string.format("%.2e",train.batch:max())..
	    ", pmn: "..string.format("%.2e",train.params:mean())..
	    ", psd: "..string.format("%.2e",train.params:std())..
	    ", pmi: "..string.format("%.2e",train.params:min())..
	    ", pmx: "..string.format("%.2e",train.params:max())..
	    ", gmn: "..string.format("%.2e",train.grads:mean())..
	    ", gsd: "..string.format("%.2e",train.grads:std())..
	    ", gmi: "..string.format("%.2e",train.grads:min())..
	    ", gmx: "..string.format("%.2e",train.grads:max())..
	    ", omn: "..string.format("%.2e",train.old_grads:mean())..
	    ", osd: "..string.format("%.2e",train.old_grads:std())..
	    ", omi: "..string.format("%.2e",train.old_grads:min())..
	    ", omx: "..string.format("%.2e",train.old_grads:max())
	 main.draw()
      end
      
      if config.main.details or config.main.debug then
	 print(msg)
      end

      main.clock.log = os.time()
   end
end

function main.testlog(test)
   if config.main.collectgarbage and math.mod(test.n,config.train_data.batch_size*config.main.collectgarbage) == 0 then
      print("Collecting garbage at n = "..test.n)
      collectgarbage()
   end
   if not config.main.details then return end
   if (os.time() - main.clock.log) >= (config.main.logtime or 1) then
      print("n: "..test.n..
	       ", l: "..string.format("%.2e",test.l)..
	       ", obj: "..string.format("%.2e",test.objective)..
	       ", dat: "..string.format("%.2e",test.time.data)..
	       ", fpp: "..string.format("%.2e",test.time.forward)..
	       ", acc: "..string.format("%.2e",test.time.accumulate))
      main.clock.log = os.time()
   end
end

-- Execute the main program
main.main()
