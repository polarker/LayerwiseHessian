local LHClassNLLCriterion, parent = torch.class('nn.LHClassNLLCriterion', 'nn.ClassNLLCriterion')

function LHClassNLLCriterion:__init()
   parent.__init(self)

   self.gradInput = {}
   self.gradInput.grad = torch.Tensor()
   self.gradInput.hessian = torch.Tensor()
end

function LHClassNLLCriterion:updateGradInput(input, target)
	local gi = self.gradInput.grad
   local hi = self.gradInput.hessian

	if input:dim() == 1 then
		local inputSize = input:size(1)
		self.gradInput.grad:resize(inputSize)
   	self.gradInput.hessian:resize(inputSize, inputSize)
   	gi:zero()
   	hi:zero()

		if torch.isTensor(target) then target = target[1] end
		gi[target] = -1
      if self.weights then
         gi[target] = gi*self.weights[target]
      end
      return self.gradInput
   elseif input:dim() == 2 then
   	local nframe = input:size(1)
      local inputSize = input:size(2)
      gi:resize(nframe, inputSize)
      hi:resize(nframe, inputSize, inputSize)

   	local z = -1
      if self.sizeAverage then
         z = z / target:size(1)
      end
      for i=1,target:size(1) do
          gi[i][target[i]] = z
         if self.weights then
             gi[i][target[i]] = gi[i][target[i]]*self.weights[target[i]]
         end
      end
   	return self.gradInput
   else
   	error('matrix or vector input expected')
   end
end