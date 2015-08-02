local LHLogSoftMax, parent = torch.class('nn.LHLogSoftMax', 'nn.LogSoftMax')

function LHLogSoftMax.__init(self)
	parent.__init(self)

	self.gradInput = {}
	self.gradInput.grad = torch.Tensor()
   self.gradInput.hessian = torch.Tensor()
end

function LHLogSoftMax:updateGradInput(input, gradOutput)
	-- Note: we assume that updateOutput() has been called
	local gi = self.gradInput.grad
   local hi = self.gradInput.hessian
   local go = gradOutput.grad
   local ho = gradOutput.hessian

	if input:dim() == 1 then
		local inputSize = input:size(1)
		gi:resize(inputSize)
   	hi:resize(inputSize, inputSize)
	
		local expOutput = torch.exp(self.output)
		local transitionMatrix = torch.ger(torch.Tensor(inputSize):fill(-1), expOutput)
		local transitionMatrixDiag = torch.Tensor(transitionMatrix:storage(), 1, inputSize, inputSize+1)
		transitionMatrixDiag:add(1)
	
   	gi:mv(transitionMatrix:t(), go)
   	hi:mm(transitionMatrix:t(), torch.mm(ho, transitionMatrix))
   	hi:addr(torch.sum(go), expOutput, expOutput)
   	local hiDiag = torch.Tensor(hi:storage(), 1, inputSize, inputSize+1)
   	hiDiag:addcmul(-1, go, expOutput)
	
   	return self.gradInput
   elseif input:dim() == 2 then
   	local nframe = input:size(1)
   	local inputSize = input:size(2)
   	gi:resize(nframe, inputSize)
      hi:resize(nframe, inputSize, inputSize)

      for i=1,nframe do
      	local expOutput = torch.exp(self.output[i])
      	local transitionMatrix = torch.ger(torch.Tensor(inputSize):fill(-1), expOutput)
      	local transitionMatrixDiag = torch.Tensor(transitionMatrix:storage(), 1, inputSize, inputSize+1)
			transitionMatrixDiag:add(1)

			gi[i]:mv(transitionMatrix:t(), go[i])
			local m = torch.mm(transitionMatrix:t(), torch.mm(ho[i], transitionMatrix))
			m:addr(torch.sum(go[i]), expOutput, expOutput)
			local mDiag = torch.Tensor(m:storage(), 1, inputSize, inputSize+1)
			mDiag:addcmul(-1, go[i], expOutput)
			hi[i]:copy(m)
		end

   	return self.gradInput
   else
   	error('vector or matrix input expected')
   end
end