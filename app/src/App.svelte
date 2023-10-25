<script>
    import { onDestroy, onMount } from "svelte";

    let isActive = true;
    let prompt = "a dog";
    let last_url;
    let last_prompt;
    let forceText = false;

    async function makePrediction() {
        if (!isActive) return;

        let input = { prompt: prompt };
        if (last_url && !forceText) {
            input["image"] = last_url;
        }
        fetch("http://localhost:5001/predictions", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ input }),
        })
            .then((r) => r.json())
            .then((data) => {
                last_url = data.output;
                last_prompt = data.input.prompt;
            })
            .then(makePrediction);
    }

    onMount(() => makePrediction());

    onDestroy(() => {
        isActive = false;
    });
</script>

<input type="text" bind:value={prompt} />
<input type="checkbox" bind:checked={forceText} /> txt2img
<br />
<img src={last_url} alt={last_prompt} />
